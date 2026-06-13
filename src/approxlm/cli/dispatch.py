from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from approxlm.application.dispatcher import run_dispatcher_config


def write_json_output(output_json: str, payload: Any) -> None:
    from approxlm.adapters.persistence.sqlite import to_jsonable

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="approxlm-dispatch",
        description="Run a sequential ApproxLM dispatcher sweep from a YAML config.",
    )
    parser.add_argument("config", help="YAML dispatcher configuration")
    parser.add_argument("--output-json", help="Write the full dispatcher result, including records, to this JSON file")
    parser.add_argument("--summary-json", help="Write only the dispatcher summary to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="Do not print progress updates to stderr")
    args = parser.parse_args()

    from approxlm.adapters.persistence.sqlite import init_db, to_jsonable

    init_db()

    def update_progress(progress: float, message: str) -> None:
        if not args.quiet:
            print(f"{int(progress * 100):3d}% {message}", file=sys.stderr)

    result = run_dispatcher_config(
        args.config,
        progress_callback=None if args.quiet else update_progress,
    )
    summary = result["summary"]

    if args.output_json:
        write_json_output(args.output_json, result)
    if args.summary_json:
        write_json_output(args.summary_json, summary)

    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
