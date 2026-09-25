#!/usr/bin/env python3
"""Prepare, submit, inspect, and fetch OpenAI Batch API grading jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from run_openai_prompt import AUDIENCE_ALIASES, METHOD_CONFIG, build_prompt, strip_code_fences


ROOT = Path(__file__).resolve().parent
BATCH_DIR = ROOT / "openai_batch"
OUTPUT_DIR = ROOT / "openai"
DEFAULT_AUDIENCES = ["cs", "doctor", "patient"]
DEFAULT_METHODS = ["unigram", "bigram"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the batch workflow."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--audiences", nargs="+", default=DEFAULT_AUDIENCES, choices=sorted(AUDIENCE_ALIASES))
    prep.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=sorted(METHOD_CONFIG))
    prep.add_argument("--n-seeds", type=int, default=3)
    prep.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5"))
    prep.add_argument("--output-prefix", default="grading_batch")

    submit = sub.add_parser("submit")
    submit.add_argument("jsonl_path")
    submit.add_argument("--api-key-env", default="OPENAI_API_KEY")

    status = sub.add_parser("status")
    status.add_argument("batch_id")
    status.add_argument("--api-key-env", default="OPENAI_API_KEY")

    fetch = sub.add_parser("fetch")
    fetch.add_argument("batch_id")
    fetch.add_argument("--api-key-env", default="OPENAI_API_KEY")

    return parser.parse_args()


def client_from_env(api_key_env: str) -> OpenAI:
    """Build an OpenAI client from an environment variable."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set")
    return OpenAI(api_key=api_key)


def iter_jobs(audiences: Iterable[str], methods: Iterable[str], n_seeds: int):
    """Yield every audience/method/seed job in deterministic order."""
    for audience in audiences:
        for method in methods:
            for seed_idx in range(1, n_seeds + 1):
                yield audience, method, seed_idx


def ensure_dirs() -> None:
    """Create output directories used by the batch workflow."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def batch_paths(prefix: str) -> tuple[Path, Path]:
    """Return the JSONL input path and manifest path for a batch prefix."""
    ensure_dirs()
    return BATCH_DIR / f"{prefix}.jsonl", BATCH_DIR / f"{prefix}.manifest.json"


def prepare_batch(audiences: list[str], methods: list[str], n_seeds: int, model: str, output_prefix: str) -> None:
    """Write the Batch API JSONL input file and a small manifest."""
    jsonl_path, manifest_path = batch_paths(output_prefix)
    manifest = {
        "model": model,
        "audiences": audiences,
        "methods": methods,
        "n_seeds": n_seeds,
        "jsonl_path": str(jsonl_path),
        "jobs": [],
    }

    with jsonl_path.open("w", encoding="utf-8") as f:
        for audience, method, seed_idx in iter_jobs(audiences, methods, n_seeds):
            prompt = build_prompt(audience, method)
            custom_id = f"{audience}_{method}_{seed_idx}"
            body = {
                "model": model,
                "input": prompt,
            }
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            manifest["jobs"].append(
                {
                    "custom_id": custom_id,
                    "audience": audience,
                    "method": method,
                    "seed": seed_idx,
                }
            )

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(jsonl_path)
    print(manifest_path)
    print(f"requests={len(manifest['jobs'])}")


def submit_batch(jsonl_path: str, api_key_env: str) -> None:
    """Upload a JSONL file and create a Batch API job."""
    client = client_from_env(api_key_env)
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing batch input file: {path}")

    uploaded = client.files.create(file=path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"source": "magenetrunner_grading", "input_file": path.name},
    )
    print(json.dumps({"file_id": uploaded.id, "batch_id": batch.id, "status": batch.status}, indent=2))


def show_status(batch_id: str, api_key_env: str) -> None:
    """Print a concise status summary for one batch."""
    client = client_from_env(api_key_env)
    batch = client.batches.retrieve(batch_id)
    payload = {
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": batch.input_file_id,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": getattr(batch, "request_counts", None),
    }
    print(json.dumps(payload, indent=2, default=str))


def response_text_from_batch_line(line: dict) -> str:
    """Extract response text from one Batch API output line."""
    body = line["response"]["body"]
    if "output_text" in body and body["output_text"]:
        return body["output_text"]

    pieces = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                pieces.append(content.get("text", ""))
    return "".join(pieces)


def fetch_batch(batch_id: str, api_key_env: str) -> None:
    """Download a completed batch output and materialize numbered JSON files."""
    client = client_from_env(api_key_env)
    batch = client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id yet. Current status: {batch.status}")

    ensure_dirs()
    output_text = client.files.content(batch.output_file_id).text
    raw_path = BATCH_DIR / f"{batch_id}.output.jsonl"
    raw_path.write_text(output_text, encoding="utf-8")

    if batch.error_file_id:
        error_text = client.files.content(batch.error_file_id).text
        (BATCH_DIR / f"{batch_id}.errors.jsonl").write_text(error_text, encoding="utf-8")

    for raw_line in output_text.splitlines():
        if not raw_line.strip():
            continue
        line = json.loads(raw_line)
        custom_id = line["custom_id"]
        if line.get("error"):
            print(f"ERROR {custom_id}: {line['error']}")
            continue

        raw_response = response_text_from_batch_line(line)
        cleaned = strip_code_fences(raw_response)
        parsed = json.loads(cleaned)
        out_path = OUTPUT_DIR / f"{custom_id}.json"
        out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out_path)

    print(raw_path)


def main() -> None:
    """Run one batch subcommand."""
    args = parse_args()
    if args.command == "prepare":
        prepare_batch(args.audiences, args.methods, args.n_seeds, args.model, args.output_prefix)
    elif args.command == "submit":
        submit_batch(args.jsonl_path, args.api_key_env)
    elif args.command == "status":
        show_status(args.batch_id, args.api_key_env)
    elif args.command == "fetch":
        fetch_batch(args.batch_id, args.api_key_env)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
