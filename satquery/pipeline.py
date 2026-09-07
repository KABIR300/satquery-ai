"""Application orchestration independent of Streamlit and any specific model implementation."""

import hashlib
import logging
import re
from time import perf_counter

import numpy as np

from satquery import __version__
from satquery.backends.base import BackendUnavailable, OutputError
from satquery.backends.change import BaselineChangeDetector
from satquery.backends.mock import MockBackend
from satquery.backends.sar import ExperimentalSARBackend
from satquery.config import Settings
from satquery.evidence import assess_quality, check_alignment, evidence_score
from satquery.geo import box_to_geo
from satquery.grounding import validate_boxes, validate_mask
from satquery.io import InputError, load_image
from satquery.routing import route
from satquery.schemas import Evidence, MaskInfo, PipelineOutput, Result, Status, TaskType

logger = logging.getLogger(__name__)
SEMANTIC_CHANGE = re.compile(
    r"\b(water|flood\w*|vegetation|forest\w*|built[- ]?up|urban|land[- ]?cover|"
    r"building\w*|road\w*|damage\w*)\b",
    re.I,
)


class Pipeline:
    def __init__(self, settings: Settings, optical_backend=None, change_backend=None, sar_backend=None):
        self.settings = settings
        if optical_backend is not None:
            self.optical = optical_backend
        elif settings.backend == "qwen":
            from satquery.backends.qwen import QwenBackend

            self.optical = QwenBackend(settings)
        else:
            self.optical = MockBackend()
        self.change = change_backend or BaselineChangeDetector(settings)
        self.sar = sar_backend or ExperimentalSARBackend()

    def run(
        self,
        question: str,
        sources: list,
        *,
        names: list[str] | None = None,
        modality="auto",
        task="auto",
        user_confirmed_alignment=False,
        window=None,
    ) -> PipelineOutput:
        started = perf_counter()
        images = []
        try:
            if (
                not isinstance(question, str)
                or not 1 <= len(question.strip()) <= self.settings.max_question_chars
            ):
                raise InputError(f"Question must contain 1–{self.settings.max_question_chars} characters.")
            if not isinstance(sources, list) or not 1 <= len(sources) <= 2:
                raise InputError("Supply one image or a before/after pair.")
            if names is not None and len(names) != len(sources):
                raise InputError("Upload filenames must match the supplied images.")
            for i, source in enumerate(sources):
                images.append(
                    load_image(
                        source,
                        self.settings,
                        name=names[i] if names else None,
                        modality=modality,
                        window=window,
                    )
                )
            output = self.analyze(
                question, images, task=task, user_confirmed_alignment=user_confirmed_alignment
            )
        except (InputError, ValueError, OSError) as exc:
            output = PipelineOutput(
                Result(
                    answer=str(exc),
                    task_type=TaskType.UNSUPPORTED,
                    status=Status.INVALID_INPUT,
                    image_metadata=[i.metadata for i in images],
                ),
                images,
            )
        output.result.latency_seconds = perf_counter() - started
        return output

    def analyze(self, question, images, *, task="auto", user_confirmed_alignment=False):
        started = perf_counter()
        selection = route(question, images, task)
        metadata = [image.metadata for image in images]
        notes = [note for image in images for note in image.warnings]
        base = dict(
            task_type=selection.task,
            image_metadata=metadata,
            warnings=notes,
            provenance={
                "application_version": __version__,
                "question": question,
                "settings": self.settings.model_dump(mode="json"),
                "synthetic_input": any(i.metadata.synthetic for i in images),
                "loaded_analysis_sha256": [
                    hashlib.sha256(np.ascontiguousarray(i.analysis).tobytes() + i.valid.tobytes()).hexdigest()
                    for i in images
                ],
            },
        )

        def finish(answer, status, **kwargs):
            mask = kwargs.pop("mask", None)
            difference = kwargs.pop("difference", None)
            valid_mask = kwargs.pop("valid_mask", None)
            record = base | kwargs
            result = Result(answer=answer, status=status, latency_seconds=perf_counter() - started, **record)
            return PipelineOutput(result, images, mask, difference, valid_mask)

        if selection.status:
            return finish(selection.reason, selection.status)
        if selection.task == TaskType.SAR:
            try:
                self.sar.analyze(images, question)
            except BackendUnavailable as exc:
                return finish(str(exc), Status.UNSUPPORTED, backend=self.sar.name)
            except Exception as exc:
                logger.exception("SAR provider failed")
                return finish(
                    f"SAR provider failed ({type(exc).__name__}); no interpretation accepted.",
                    Status.INVALID_OUTPUT,
                    backend=self.sar.name,
                )
            return finish(
                "SAR provider output validation is not implemented in this MVP.",
                Status.UNSUPPORTED,
                backend=self.sar.name,
            )
        qualities = [assess_quality(i) for i in images]
        for q in qualities:
            notes.extend(q.warnings)
        base["provenance"]["input_quality"] = [q.signals for q in qualities]
        if any(not q.usable for q in qualities):
            return finish(
                "Image quality is insufficient. Select a clearer, valid tile before analysis.",
                Status.NEED_BETTER_INPUT,
                evidence=Evidence(score=0, signals={"quality_gate": False}),
            )
        backend = self.change if selection.task == TaskType.CHANGE_DETECTION else self.optical
        try:
            if selection.task == TaskType.CHANGE_DETECTION:
                alignment = check_alignment(*images, self.settings, user_confirmed=user_confirmed_alignment)
                notes.extend(alignment.warnings)
                if not alignment.accepted:
                    return finish(
                        alignment.warnings[-1],
                        Status.NEED_BETTER_INPUT,
                        backend=backend.name,
                        evidence=Evidence(score=0, signals=alignment.signals),
                    )
                prediction = backend.analyze(*images)
                shape = images[0].valid.shape
                validate_mask(prediction.mask, shape, prediction.valid)
                if not np.array_equal(prediction.valid, images[0].valid & images[1].valid):
                    raise OutputError("Change backend changed the valid data footprint.")
                if (
                    prediction.difference.shape != shape
                    or not np.isfinite(prediction.difference).all()
                    or np.any(prediction.difference < 0)
                    or np.any(prediction.difference > 1)
                ):
                    raise OutputError("Difference map must be finite, in [0,1], and match the image grid.")
                validate_boxes(prediction.boxes, shape[1], shape[0])
                notes.extend(prediction.warnings)
                evidence = evidence_score(images, qualities, spatial_valid=True, alignment=alignment)
                changed, count = int(prediction.mask.sum()), int(prediction.valid.sum())
                fraction = changed / count
                answer = (
                    f"Baseline appearance change covers {fraction:.1%} of shared valid preview pixels "
                    f"({changed:,} of {count:,})."
                )
                status = (
                    Status.OK if evidence.score >= self.settings.evidence_threshold else Status.LOW_EVIDENCE
                )
                if SEMANTIC_CHANGE.search(question):
                    answer += " This baseline cannot identify water, vegetation, built-up classes or their direction of change."
                    status = Status.LOW_EVIDENCE
                if alignment.score < 0.8:
                    status = Status.LOW_EVIDENCE
                if fraction > 0.8:
                    notes.append("Most pixels differ; radiometric or registration mismatch may dominate.")
                    status = Status.LOW_EVIDENCE
                info = MaskInfo(
                    name="appearance-change",
                    width=shape[1],
                    height=shape[0],
                    positive_pixels=changed,
                    valid_pixels=count,
                )
                output = finish(
                    answer,
                    status,
                    evidence=evidence,
                    backend=prediction.backend,
                    bounding_boxes=prediction.boxes,
                    masks=[info],
                    mask=prediction.mask,
                    difference=prediction.difference,
                    valid_mask=prediction.valid,
                    statistics=prediction.statistics,
                )
            else:
                prediction = backend.analyze(images[0], question)
                h, w = images[0].valid.shape
                validate_boxes(prediction.boxes, w, h)
                if not isinstance(prediction.answer, str) or not prediction.answer.strip():
                    raise OutputError("Backend returned an empty answer.")
                notes.extend(prediction.warnings)
                if not prediction.boxes:
                    return finish(
                        "No valid spatial evidence supports an optical answer. Try a clearer tile "
                        "or a more specific visible feature.",
                        Status.LOW_EVIDENCE,
                        backend=prediction.backend,
                        evidence=Evidence(score=0, signals={"spatial_evidence_valid": False}),
                    )
                # A box over nodata cannot support an answer.
                for box in prediction.boxes:
                    region = images[0].valid[
                        int(box.y1) : int(np.ceil(box.y2)), int(box.x1) : int(np.ceil(box.x2))
                    ]
                    if not region.size or region.mean() < 0.5:
                        raise OutputError("A proposed box is predominantly outside valid image data.")
                evidence = evidence_score(
                    images, qualities, spatial_valid=True, demonstration=prediction.demonstration
                )
                base["provenance"].update(prediction.provenance)
                if not prediction.demonstration:
                    notes.append(
                        "Uncalibrated VLM answer and proposed boxes require expert review. "
                        "Valid coordinates do not verify semantic correctness."
                    )
                output = finish(
                    prediction.answer,
                    Status.LOW_EVIDENCE,
                    backend=prediction.backend,
                    evidence=evidence,
                    bounding_boxes=prediction.boxes,
                )
            target = images[-1].metadata
            if target.crs:
                for box in output.result.bounding_boxes:
                    try:
                        output.result.geospatial.append(box_to_geo(box, target))
                    except ValueError:
                        output.result.warnings.append(
                            "A box could not be reprojected; pixel evidence is retained."
                        )
            return output
        except OutputError as exc:
            return finish(f"Model output rejected: {exc}", Status.INVALID_OUTPUT, backend=backend.name)
        except BackendUnavailable as exc:
            return finish(str(exc), Status.UNSUPPORTED, backend=backend.name)
        except Exception as exc:
            logger.exception("Analysis backend failed")
            return finish(
                f"Analysis failed ({type(exc).__name__}). No answer was accepted. "
                "See the application log for diagnostics.",
                Status.INVALID_OUTPUT,
                backend=backend.name,
            )
