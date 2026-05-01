"""Command line interface for lobpy."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__
from .validation import ParquetValidationResult, validate_parquet


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``lobpy`` argument parser."""
    parser = argparse.ArgumentParser(prog="lobpy", description="lobpy command line utilities")
    parser.add_argument("--version", action="version", version=f"lobpy {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate lobpy parquet event files",
        description="Validate parquet files against the lobpy TL.from_parquet data contract.",
    )
    validate_parser.add_argument("paths", nargs="+", help="parquet file(s) to validate")
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="require the full documented schema and strict nullable/type metadata",
    )
    validate_parser.add_argument(
        "--load",
        action="store_true",
        help="also try loading each file with TL.from_parquet(..., mode='lazy')",
    )
    validate_parser.add_argument(
        "--full-check",
        action="store_true",
        help="replay events in eager mode and run lob.check() after each timestamp",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON",
    )
    validate_parser.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="maximum row indices to show per issue (default: 5)",
    )
    validate_parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="return a non-zero exit code when warnings are present",
    )
    validate_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only print invalid files in human-readable mode",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lobpy command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        results = [
            validate_parquet(path, strict=args.strict, load=args.load,
                             full_check=args.full_check, max_rows=args.max_rows)
            for path in args.paths
        ]
        if args.as_json:
            payload = [result.to_dict() for result in results]
            print(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2))
        else:
            for result in results:
                if args.quiet and result.ok and not result.warnings:
                    continue
                _print_result(result)

        failed = any(not result.ok for result in results)
        warned = any(result.warnings for result in results)
        return 1 if failed or (args.warnings_as_errors and warned) else 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _print_result(result: ParquetValidationResult) -> None:
    label = "OK" if result.ok else "ERROR"
    row_info = "unknown rows" if result.rows is None else f"{result.rows} rows"
    elapsed = f"{result.elapsed:.3f}s" if result.elapsed is not None else "?"
    print(
        f"{label}: {result.path} ({row_info}, "
        f"{len(result.errors)} error(s), {len(result.warnings)} warning(s), "
        f"{elapsed})"
    )

    for issue in result.errors + result.warnings:
        rows = f" rows={issue.rows}" if issue.rows else ""
        print(f"  [{issue.severity.upper()}] {issue.code}: {issue.message}{rows}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
