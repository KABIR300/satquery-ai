import argparse
import json
import sys
from pathlib import Path

from satquery.config import load_settings


def main():
    # Pipes use a legacy code page on some Windows installs; exports and CLI JSON use UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SatQuery AI — evidence-first remote-sensing MVP")
    sub = parser.add_subparsers(dest="command", required=True)
    hardware = sub.add_parser("hardware", help="Inspect CPU, RAM, NVIDIA and optional PyTorch")
    hardware.add_argument("--no-torch", action="store_true")
    samples = sub.add_parser("samples", help="Generate a tiny synthetic fixture pack")
    samples.add_argument("--output", default="data/samples")
    analyze = sub.add_parser("analyze", help="Run a single analysis and export evidence")
    analyze.add_argument("images", nargs="+")
    analyze.add_argument("--question", required=True)
    analyze.add_argument("--config")
    analyze.add_argument("--backend", choices=["mock", "qwen"])
    analyze.add_argument("--task", choices=["auto", "optical", "change"], default="auto")
    analyze.add_argument("--modality", choices=["auto", "optical", "sar"], default="auto")
    analyze.add_argument("--window", type=int, nargs=4, metavar=("COL", "ROW", "WIDTH", "HEIGHT"))
    analyze.add_argument("--confirm-alignment", action="store_true")
    analyze.add_argument("--output", default="output/evidence.zip")
    evaluation = sub.add_parser("evaluate", help="Run an annotated JSONL pack")
    evaluation.add_argument("manifest")
    evaluation.add_argument("--config")
    evaluation.add_argument("--output", default="output/evaluation.json")
    args = parser.parse_args()
    if args.command == "hardware":
        from satquery.hardware import inspect_hardware

        print(json.dumps(inspect_hardware(not args.no_torch).as_dict(), indent=2))
        return 0
    if args.command == "samples":
        from satquery.samples import generate_samples

        print(f"Synthetic fixtures written: {generate_samples(args.output)}")
        return 0
    try:
        from satquery.pipeline import Pipeline

        settings = load_settings(args.config, backend=getattr(args, "backend", None))
        pipeline = Pipeline(settings)
        if args.command == "evaluate":
            from satquery.evaluation import evaluate, save_report

            report = evaluate(args.manifest, pipeline)
            save_report(report, args.output)
            print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
        else:
            from satquery.export import evidence_bundle

            output = pipeline.run(
                args.question,
                args.images,
                modality=args.modality,
                task=args.task,
                user_confirmed_alignment=args.confirm_alignment,
                window=tuple(args.window) if args.window else None,
            )
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(evidence_bundle(output))
            print(output.result.model_dump_json(indent=2))
            return 0 if output.result.status.value in {"OK", "LOW_EVIDENCE"} else 2
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0
