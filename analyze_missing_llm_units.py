#!/usr/bin/env python3
"""Analyze missing unigram and bigram LLM scores against canonical unit lists."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FILENAME_RE = re.compile(
    r"^(?P<audience>.+)_(?P<gram>unigram|bigram)(?:_lib(?P<library>\d+))?_(?P<seed>\d+)\.json$"
)
UNIGRAM_ROW_RE = re.compile(r"^\|\s*([^\|]+?)\s*\|")
LIBRARY_RE = re.compile(r"^#+\s+Library\s+(\d+)\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(ROOT))
    parser.add_argument("--llms", nargs="+", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "analysis"))
    parser.add_argument("--audiences", nargs="*", default=None)
    parser.add_argument("--grams", nargs="*", choices=("unigram", "bigram"), default=None)
    return parser.parse_args()


def parse_library_id(raw_value: Any) -> int:
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, str):
        match = re.search(r"(\d+)", raw_value)
        if match:
            return int(match.group(1))
    raise ValueError(f"Unsupported library_id value: {raw_value!r}")


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise TypeError(f"Unexpected JSON payload type: {type(payload)!r}")


def load_score_file(path: Path) -> list[dict[str, Any]]:
    return normalize_payload(json.loads(path.read_text(encoding="utf-8")))


def load_unigram_keys(path: Path) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    current_library: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        lib_match = LIBRARY_RE.match(line)
        if lib_match:
            current_library = int(lib_match.group(1))
            continue
        if current_library is None or not line.startswith("|"):
            continue
        match = UNIGRAM_ROW_RE.match(line)
        if not match:
            continue
        unit = match.group(1).strip()
        if unit in {"Fn", "---"}:
            continue
        keys.add((current_library, unit))
    return keys


def load_bigram_keys(path: Path) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    current_library: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lib_match = LIBRARY_RE.match(line)
        if lib_match:
            current_library = int(lib_match.group(1))
            continue
        if current_library is None or line.startswith("#"):
            continue
        keys.add((current_library, line))
    return keys


def canonical_keys() -> dict[str, set[tuple[int, str]]]:
    return {
        "unigram": load_unigram_keys(ROOT / "symbol_descriptions.md"),
        "bigram": load_bigram_keys(ROOT / "symbol_compositions.md"),
    }


def discover_score_files(
    llm_dir: Path,
    allowed_audiences: set[str] | None,
    allowed_grams: set[str] | None,
) -> dict[tuple[str, str], list[Path]]:
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(llm_dir.glob("*.json")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        audience = match.group("audience")
        gram = match.group("gram")
        if allowed_audiences is not None and audience not in allowed_audiences:
            continue
        if allowed_grams is not None and gram not in allowed_grams:
            continue
        grouped[(audience, gram)].append(path)
    return grouped


def score_keys_from_file(path: Path) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for library_block in load_score_file(path):
        library_id = parse_library_id(library_block["library_id"])
        for entry in library_block["scores"]:
            keys.add((library_id, str(entry["unit"])))
    return keys


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    allowed_audiences = set(args.audiences) if args.audiences else None
    allowed_grams = set(args.grams) if args.grams else None
    expected_by_gram = canonical_keys()

    per_file_rows: list[dict[str, Any]] = []
    per_group_rows: list[dict[str, Any]] = []
    missing_detail_rows: list[dict[str, Any]] = []

    for llm_name in args.llms:
        llm_dir = input_root / llm_name
        if not llm_dir.is_dir():
            raise FileNotFoundError(f"Missing LLM directory: {llm_dir}")

        groups = discover_score_files(llm_dir, allowed_audiences, allowed_grams)
        for (audience, gram), paths in sorted(groups.items()):
            expected = expected_by_gram[gram]
            group_union: set[tuple[int, str]] = set()
            coverage_counts: dict[tuple[int, str], int] = defaultdict(int)

            for path in paths:
                scored = score_keys_from_file(path)
                missing = sorted(expected - scored)
                group_union |= scored
                for key in scored:
                    if key in expected:
                        coverage_counts[key] += 1
                match = FILENAME_RE.match(path.name)
                assert match is not None
                seed = int(match.group("seed"))
                per_file_rows.append(
                    {
                        "llm": llm_name,
                        "audience": audience,
                        "gram": gram,
                        "seed": seed,
                        "expected_units": len(expected),
                        "scored_units": len(scored & expected),
                        "missing_units": len(missing),
                        "coverage_pct": round(100.0 * len(scored & expected) / len(expected), 2),
                        "path": str(path),
                    }
                )
                for library_id, unit in missing:
                    missing_detail_rows.append(
                        {
                            "llm": llm_name,
                            "audience": audience,
                            "gram": gram,
                            "seed": seed,
                            "library_id": library_id,
                            "unit": unit,
                            "path": str(path),
                        }
                    )

            never_scored = sorted(expected - group_union)
            missing_in_some = sum(1 for key in expected if coverage_counts[key] != len(paths))
            per_group_rows.append(
                {
                    "llm": llm_name,
                    "audience": audience,
                    "gram": gram,
                    "n_files": len(paths),
                    "expected_units": len(expected),
                    "covered_union_units": len(group_union & expected),
                    "never_scored_units": len(never_scored),
                    "missing_in_some_files_units": missing_in_some,
                    "complete_in_all_files_units": len(expected) - missing_in_some,
                    "group_union_coverage_pct": round(100.0 * len(group_union & expected) / len(expected), 2),
                }
            )
            print(
                f"[{llm_name}] {audience} {gram}: expected={len(expected)} "
                f"union_covered={len(group_union & expected)} never_scored={len(never_scored)} "
                f"missing_in_some={missing_in_some}"
            )

    write_csv(
        output_dir / "llm_missing_units_per_file.csv",
        per_file_rows,
        [
            "llm",
            "audience",
            "gram",
            "seed",
            "expected_units",
            "scored_units",
            "missing_units",
            "coverage_pct",
            "path",
        ],
    )
    write_csv(
        output_dir / "llm_missing_units_per_group.csv",
        per_group_rows,
        [
            "llm",
            "audience",
            "gram",
            "n_files",
            "expected_units",
            "covered_union_units",
            "never_scored_units",
            "missing_in_some_files_units",
            "complete_in_all_files_units",
            "group_union_coverage_pct",
        ],
    )
    write_csv(
        output_dir / "llm_missing_units_detail.csv",
        missing_detail_rows,
        ["llm", "audience", "gram", "seed", "library_id", "unit", "path"],
    )
    print(output_dir / "llm_missing_units_per_group.csv")
    print(output_dir / "llm_missing_units_per_file.csv")
    print(output_dir / "llm_missing_units_detail.csv")


if __name__ == "__main__":
    main()
