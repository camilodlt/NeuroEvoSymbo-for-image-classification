#!/usr/bin/env python3
"""Aggregate LLM score JSON files by audience and gram."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FILENAME_RE = re.compile(
    r"^(?P<audience>.+)_(?P<gram>unigram|bigram)(?:_lib(?P<library>\d+))?_(?P<seed>\d+)\.json$"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for selecting LLM folders and output locations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default=str(ROOT),
        help="Directory that contains one folder per LLM, such as chatgpt/ or gemini/.",
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        required=True,
        help="LLM subfolders under input-root to include, for example: chatgpt gemini.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "aggregated"),
        help="Directory where aggregated per-LLM and global JSON files will be written.",
    )
    parser.add_argument(
        "--audiences",
        nargs="*",
        default=None,
        help="Optional audience filter, for example: cs doctor patient.",
    )
    parser.add_argument(
        "--grams",
        nargs="*",
        default=None,
        choices=("unigram", "bigram"),
        help="Optional gram filter.",
    )
    return parser.parse_args()


def parse_library_id(raw_value: Any) -> int:
    """Extract the numeric library id from an integer-like or string label."""
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, str):
        match = re.search(r"(\d+)", raw_value)
        if match:
            return int(match.group(1))
    raise ValueError(f"Unsupported library_id value: {raw_value!r}")


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize a score payload into a list of per-library score objects."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise TypeError(f"Unexpected JSON payload type: {type(payload)!r}")


def load_score_file(path: Path) -> list[dict[str, Any]]:
    """Load a single JSON score file and normalize its top-level shape."""
    with path.open("r", encoding="utf-8") as handle:
        return normalize_payload(json.load(handle))


def discover_score_files(
    llm_dir: Path,
    allowed_audiences: set[str] | None,
    allowed_grams: set[str] | None,
) -> dict[tuple[str, str], list[Path]]:
    """Collect score JSON files from one LLM folder grouped by audience and gram."""
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


def aggregate_group(paths: list[Path]) -> list[dict[str, Any]]:
    """Average scores over a list of JSON files while preserving library grouping."""
    score_sums: dict[tuple[int, str], float] = defaultdict(float)
    score_counts: dict[tuple[int, str], int] = defaultdict(int)
    unit_order: list[tuple[int, str]] = []
    seen_units: set[tuple[int, str]] = set()
    file_presence: dict[tuple[int, str], set[Path]] = defaultdict(set)

    for path in paths:
        for library_block in load_score_file(path):
            library_id = parse_library_id(library_block["library_id"])
            for entry in library_block["scores"]:
                unit = str(entry["unit"])
                key = (library_id, unit)
                if key not in seen_units:
                    seen_units.add(key)
                    unit_order.append(key)
                score_sums[key] += float(entry["score"])
                score_counts[key] += 1
                file_presence[key].add(path)

    expected = len(paths)
    for library_id, unit in unit_order:
        key = (library_id, unit)
        present = len(file_presence[key])
        if present != expected:
            missing_files = sorted(str(path) for path in set(paths) - file_presence[key])
            print(
                f"[warn] missing score in some files: library={library_id} unit={unit} "
                f"present={present}/{expected} missing_files={missing_files}"
            )

    per_library_scores: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for library_id, unit in unit_order:
        key = (library_id, unit)
        mean_score = score_sums[key] / score_counts[key]
        per_library_scores[library_id].append(
            {
                "unit": unit,
                "score": mean_score,
                "rationale": f"Average over {score_counts[key]} runs.",
            }
        )

    output: list[dict[str, Any]] = []
    for library_id in sorted(per_library_scores):
        output.append(
            {
                "library_id": f"Library {library_id}",
                "scores": per_library_scores[library_id],
            }
        )
    return output


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write a JSON file with a stable indentation and UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    """Aggregate per-LLM and global score files for the selected audiences and grams."""
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    allowed_audiences = set(args.audiences) if args.audiences else None
    allowed_grams = set(args.grams) if args.grams else None

    global_groups: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for llm_name in args.llms:
        llm_dir = input_root / llm_name
        if not llm_dir.is_dir():
            raise FileNotFoundError(f"Missing LLM directory: {llm_dir}")

        groups = discover_score_files(llm_dir, allowed_audiences, allowed_grams)
        for (audience, gram), paths in sorted(groups.items()):
            print(f"[per-llm] {llm_name} {audience} {gram}: {len(paths)} file(s)")
            payload = aggregate_group(paths)
            out_path = output_root / llm_name / f"{audience}_{gram}_avg.json"
            write_json(out_path, payload)
            print(out_path)
            global_groups[(audience, gram)].extend(paths)

    for (audience, gram), paths in sorted(global_groups.items()):
        print(f"[global] {audience} {gram}: {len(paths)} file(s)")
        payload = aggregate_group(paths)
        out_path = output_root / "global" / f"{audience}_{gram}_avg.json"
        write_json(out_path, payload)
        print(out_path)


if __name__ == "__main__":
    main()
