"""REAL CPU baseline: common-scale absolute band difference, morphology, components."""

import cv2
import numpy as np

from satquery.backends.base import BackendUnavailable, ChangePrediction
from satquery.config import Settings
from satquery.schemas import Box


class BaselineChangeDetector:
    name = "baseline/absolute-difference-v1"

    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, before, after):
        valid = before.valid & after.valid
        a, b = before.analysis, after.analysis
        # Both observations share each band's scale. Independent image stretches would invent change.
        values = np.concatenate([a[valid], b[valid]], axis=0)
        low, high = np.percentile(values, [2, 98], axis=0)
        minimum, maximum = values.min(axis=0), values.max(axis=0)
        low = np.where(high > low, low, minimum)
        high = np.where(high > low, high, maximum)
        scale = np.maximum(high - low, 1e-6)
        difference = np.clip(np.abs(a - b) / scale, 0, 1).mean(axis=2).astype(np.float32)
        difference[~valid] = 0
        mask = (difference >= self.settings.change_threshold) & valid
        # The mean difference is NOT a learned probability.
        if self.settings.min_region_pixels >= 9:
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(
                bool
            )
        mask &= valid
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= self.settings.min_region_pixels]
        lookup = np.zeros(count, bool)
        lookup[keep] = True
        mask = lookup[labels] & valid
        boxes = []
        ordered = sorted(keep, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
        for idx in ordered[:100]:
            x, y, w, h, _ = [int(v) for v in stats[idx]]
            boxes.append(Box(x1=x, y1=y, x2=x + w, y2=y + h, label="Appearance change"))
        changed, valid_count = int(mask.sum()), int(valid.sum())
        notes = [
            "Baseline appearance difference; not a neural detector, change probability, or land-cover map.",
            "Lighting, clouds, seasonality and registration errors can also cause differences.",
        ]
        if len(keep) > 100:
            notes.append("Only the largest 100 region boxes are displayed; the mask retains all regions.")
        return ChangePrediction(
            mask,
            difference,
            valid,
            boxes,
            self.name,
            {
                "changed_pixels": changed,
                "valid_pixels": valid_count,
                "changed_fraction": changed / valid_count,
                "region_count": len(keep),
                "difference_threshold": self.settings.change_threshold,
                "normalization": "Shared before/after per-band 2–98 percentile range",
                "analysis_resolution": f"{mask.shape[1]} x {mask.shape[0]} preview pixels",
            },
            notes,
        )


class LearnedChangeDetector:
    name = "learned-change/interface-only"

    def analyze(self, before, after):
        raise BackendUnavailable("Learned change detection is planned; no trained checkpoint is provided.")
