"""
Command-line interface for running the GenConvo pipeline.

Entry point defined in pyproject:
  [project.scripts]
  genconvo = "genconvo.cli:main"
"""

from __future__ import annotations

import argparse
import json
import sys

from .synthesizer import GenConvoSynthesizer
from .data.finance import FINANCE_BENCH_PATH


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genconvo",
        description="Run GenConvoBench pipeline for a FinanceBench document and export results",
    )

    parser.add_argument(
        "doc_name",
        type=str,
        help="FinanceBench document name, e.g., 'AMD_2022_10K'",
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        default=16,
        help="Number of questions to generate (default: 16)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model name for generation (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel workers for the pipeline (default: 8)",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="factual",
        choices=["factual", "knowledge", "disjoint", "synthesized", "structured", "creative", "counting", "reasoning"],
        help="Prompt type for question generation (default: factual)",
    )
    parser.add_argument(
        "--warmup",
        nargs="?",
        const="on",
        choices=["on", "force"],
        help=(
            "Run with a default warmup config (factual, 1 question, 1 worker, model claude-sonnet-4-20250514, temperature 0.7). "
            "If provided, these settings override --num-questions/--max-workers/--model-name/--temperature/--prompt-type."
        ),
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print a compact JSON summary to stdout",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        # Hardcode FinanceBench dataset path
        dataset_directory = str(FINANCE_BENCH_PATH)
        filename = f"{args.doc_name}.md"

        args_dict = {
            "num_questions": args.num_questions,
            "max_workers": args.max_workers,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "prompt_type": args.prompt_type,
        }

        # Warmup overrides
        if args.warmup is not None:
            overridden_flags = [
                flag for flag in ["--num-questions", "--max-workers", "--model-name", "--temperature", "--prompt-type"]
                if flag in sys.argv[1:]
            ]
            if overridden_flags:
                print(
                    "Warning: --warmup overrides these flags: " + ", ".join(overridden_flags),
                    file=sys.stderr,
                )
            args_dict["num_questions"] = 1
            args_dict["max_workers"] = 1


        synthesizer = GenConvoSynthesizer(
            dataset_directory=dataset_directory,
            filename=filename,
            **args_dict
        )

        results = synthesizer()

        summary = {
            "dataset_path": results.get("dataset_path"),
            "total_questions": results.get("total_questions"),
            "context": results.get("context"),
        }

        if args.print_json:
            print(json.dumps(summary))
        else:
            dp = summary.get("dataset_path")
            tq = summary.get("total_questions")
            if args.warmup:
                print("Warmup run complete.")
            print(f"Saved dataset to: {dp}")
            print(f"Total questions: {tq}")

        return 0
    except Exception as exc:  # pragma: no cover - CLI robustness
        print(f"Error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())


