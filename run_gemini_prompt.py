#!/usr/bin/env python3
"""Compose the LLM grading prompt and save Gemini's JSON response."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from google import genai


ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
AUDIENCES_DIR = PROMPTS_DIR / "audiences"
SCORING_DIR = PROMPTS_DIR / "scoring"
OUTPUT_DIR = ROOT / "gemini"

AUDIENCE_ALIASES = {
    "cs": "computer_scientist",
    "computer_scientist": "computer_scientist",
    "doctor": "doctor",
    "patient": "patient",
}

METHOD_CONFIG = {
    "unigram": {
        "scoring_file": SCORING_DIR / "unigram.txt",
        "units_file": ROOT / "symbol_descriptions.md",
    },
    "bigram": {
        "scoring_file": SCORING_DIR / "bigrams.txt",
        "units_file": ROOT / "symbol_compositions.md",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for audience, method, and model selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("audience", choices=sorted(AUDIENCE_ALIASES))
    parser.add_argument("method", choices=sorted(METHOD_CONFIG))
    parser.add_argument("--library-id", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    return parser.parse_args()


def read_text(path: Path) -> str:
    """Read a UTF-8 text file and fail with a clear message if missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def library_block(text: str, library_id: int) -> str:
    """Extract one '# Library N' block from the compositions markdown file."""
    pattern = re.compile(
        rf"(?ms)^# Library {library_id}\n+.*?(?=^# Library \d+\n|\Z)"
    )
    match = pattern.search(text.strip())
    if not match:
        raise ValueError(f"Could not find Library {library_id} block in compositions file")
    return match.group(0).strip()


def build_prompt(audience: str, method: str, library_id: int | None = None) -> str:
    """Concatenate audience, scoring, and units prompts with blank lines."""
    audience_key = AUDIENCE_ALIASES[audience]
    method_cfg = METHOD_CONFIG[method]
    units_text = read_text(method_cfg["units_file"])
    if method == "bigram":
        if library_id is None:
            raise ValueError("--library-id is required for bigram")
        units_text = "\n\n".join([read_text(ROOT / "symbol_descriptions.md"), library_block(units_text, library_id)])
    elif library_id is not None:
        raise ValueError("--library-id is only valid for bigram")
    pieces = [
        read_text(AUDIENCES_DIR / f"{audience_key}.txt"),
        read_text(method_cfg["scoring_file"]),
        units_text,
    ]
    return "\n\n".join(pieces) + "\n"


def strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown JSON code fence if Gemini emits one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def call_gemini(prompt: str, model: str, api_key: str) -> str:
    """Send the prompt to Gemini and return the raw text response."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def next_output_path(audience: str, method: str, library_id: int | None = None) -> Path:
    """Return the next numbered output path for the given audience/method pair."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lib_part = f"_lib{library_id}" if library_id is not None else ""
    prefix = f"{audience}_{method}{lib_part}_"
    existing = sorted(OUTPUT_DIR.glob(f"{prefix}*.json"))
    max_idx = 0
    for path in existing:
        stem = path.stem
        suffix = stem.removeprefix(prefix)
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return OUTPUT_DIR / f"{prefix}{max_idx + 1}.json"


def main() -> None:
    """Compose the prompt, call Gemini, and save validated JSON output."""
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")

    if args.method == "bigram" and args.library_id is None:
        merged = []
        for library_id in [1, 2, 3, 4]:
            prompt = build_prompt(args.audience, args.method, library_id)
            print(f"===== LIBRARY {library_id} =====")
            print("===== PROMPT BEGIN =====")
            print(prompt)
            print("===== PROMPT END =====")
            raw_text = call_gemini(prompt, args.model, api_key)
            print("===== RESPONSE BEGIN =====")
            print(raw_text)
            print("===== RESPONSE END =====")
            cleaned = strip_code_fences(raw_text)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                merged.append(parsed)
            else:
                merged.extend(parsed)

        out_path = next_output_path(args.audience, args.method, None)
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out_path)
        return

    library_ids = [args.library_id] if args.library_id is not None else [None]
    for library_id in library_ids:
        prompt = build_prompt(args.audience, args.method, library_id)
        if library_id is not None:
            print(f"===== LIBRARY {library_id} =====")
        print("===== PROMPT BEGIN =====")
        print(prompt)
        print("===== PROMPT END =====")
        raw_text = call_gemini(prompt, args.model, api_key)
        print("===== RESPONSE BEGIN =====")
        print(raw_text)
        print("===== RESPONSE END =====")
        cleaned = strip_code_fences(raw_text)
        parsed = json.loads(cleaned)

        out_path = next_output_path(args.audience, args.method, library_id)
        out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out_path)


if __name__ == "__main__":
    main()
