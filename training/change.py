"""Documented learned change-detector checkpoint/data contract, no fake trainer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChangeTrainingPlan:
    inputs: str = "Co-registered before/after optical bands; joint validity mask; binary change labels"
    split: str = "Split by geographic scene and event, never neighboring tiles across train/test"
    candidate: str = "Siamese U-Net or another justified learned architecture; not implemented/trained"
    checkpoint_contract: str = (
        "Band order/scales, normalization, nodata policy, CRS grid, threshold, model version"
    )
    required_validation: str = (
        "Held-out mask IoU/F1, no-change false positives, cloud and alignment failure cases"
    )


if __name__ == "__main__":
    import json
    from dataclasses import asdict

    print(json.dumps(asdict(ChangeTrainingPlan()), indent=2))
