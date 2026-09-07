"""Deterministic routing; a pair cannot be silently reduced to single-image QA."""

import re
from dataclasses import dataclass

from satquery.schemas import LoadedImage, Status, TaskType

CHANGE_WORDS = re.compile(
    r"\b(chang\w*|compar\w*|before|after|between|increas\w*|decreas\w*|flood\w*|"
    r"new|lost|gain\w*|loss|difference\w*|expand\w*|shrink\w*|spread\w*)\b",
    re.I,
)


@dataclass(frozen=True)
class Route:
    task: TaskType
    status: Status | None = None
    reason: str = ""


def route(question: str, images: list[LoadedImage], task: str = "auto") -> Route:
    if not question.strip() or not images or len(images) > 2:
        return Route(TaskType.UNSUPPORTED, Status.INVALID_INPUT, "Supply a question and one or two images.")
    if any(i.metadata.modality == "sar" for i in images):
        return Route(TaskType.SAR)
    if task not in {"auto", "optical", "change"}:
        return Route(TaskType.UNSUPPORTED, Status.INVALID_INPUT, "Unknown task selection.")
    wants_change = task == "change" or (task == "auto" and bool(CHANGE_WORDS.search(question)))
    if wants_change and len(images) != 2:
        return Route(
            TaskType.CHANGE_DETECTION,
            Status.NEED_BETTER_INPUT,
            "Supply both before and after images for a temporal question.",
        )
    if len(images) == 2 and not wants_change:
        return Route(
            TaskType.UNSUPPORTED,
            Status.UNSUPPORTED,
            "Two-image inputs support change analysis. Select Compare or ask a change question.",
        )
    return Route(TaskType.CHANGE_DETECTION if wants_change else TaskType.OPTICAL_QA)
