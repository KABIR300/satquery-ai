"""LoRA preflight and adapter factory. A validated multimodal trainer remains planned."""

import argparse
import json
from pathlib import Path


def validate_manifest(path: str | Path):
    root = Path(path).parent
    records = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(
            isinstance(row.get(k), str) and row[k].strip()
            for k in ("image", "question", "answer", "scene_id")
        ):
            raise ValueError(f"Row {n} requires image/question/answer/scene_id strings.")
        if not (root / row["image"]).is_file():
            raise ValueError(f"Row {n}: image does not exist.")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"Row {n}: explicit train/validation/test split required.")
        records.append(row)
    if not records:
        raise ValueError("Training manifest is empty.")
    split_by_scene = {}
    for row in records:
        if row["scene_id"] in split_by_scene and split_by_scene[row["scene_id"]] != row["split"]:
            raise ValueError("Scene leakage: the same scene occurs in multiple splits.")
        split_by_scene[row["scene_id"]] = row["split"]
    return {
        "record_count": len(records),
        "scene_count": len(split_by_scene),
        "splits": {
            split: sum(row["split"] == split for row in records) for split in ("train", "validation", "test")
        },
        "status": "PREFLIGHT ONLY — no training performed",
    }


def attach_lora(model, rank=8, alpha=16, dropout=0.05):
    """Explicit opt-in factory for an already-loaded model; freezes pretrained weights via PEFT."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def main():
    parser = argparse.ArgumentParser(description="Validate future LoRA data; never starts training.")
    parser.add_argument("manifest")
    args = parser.parse_args()
    try:
        print(json.dumps(validate_manifest(args.manifest), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
