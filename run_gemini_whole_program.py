#!/usr/bin/env python3
"""Send one exported full program to Gemini for stakeholder-specific interpretation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from google import genai


ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
AUDIENCES_DIR = PROMPTS_DIR / "audiences"
OUTPUT_DIR = ROOT / "gemini_whole_program"

AUDIENCE_ALIASES = {
    "cs": "computer_scientist",
    "computer_scientist": "computer_scientist",
    "doctor": "doctor",
    "patient": "patient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audience", choices=sorted(AUDIENCE_ALIASES))
    parser.add_argument("--program-path", required=True, help="Path to one exported .jl program file")
    parser.add_argument(
        "--image-paths",
        nargs="*",
        default=(),
        help="Optional image file paths to send alongside the program prompt.",
    )
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8").strip()


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def build_prompt(audience: str, program_path: Path) -> str:
    audience_key = AUDIENCE_ALIASES[audience]
    pieces = [
        read_text(AUDIENCES_DIR / f"{audience_key}.txt"),
        read_text(PROMPTS_DIR / "whole_program.txt"),
        read_text(ROOT / "symbol_descriptions.md"),
        "Program:\n" + read_text(program_path),
    ]
    return "\n\n".join(pieces) + "\n"


def _mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Unsupported image suffix for Gemini upload: {path}")


def call_gemini(prompt: str, model: str, api_key: str, image_paths: list[Path]) -> str:
    client = genai.Client(api_key=api_key)
    contents: list[object] = [prompt]
    for image_path in image_paths:
        contents.append(
            genai.types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type=_mime_type_for_path(image_path),
            )
        )
    response = client.models.generate_content(model=model, contents=contents)
    return response.text or ""


def next_output_path(output_dir: Path, audience: str, program_path: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = program_path.stem
    prefix = f"{audience}_{stem}_"
    max_idx = 0
    for path in sorted(output_dir.glob(f"{prefix}*.json")):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return output_dir / f"{prefix}{max_idx + 1}.json"


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")

    program_path = Path(args.program_path).resolve()
    if program_path.suffix != ".jl":
        raise ValueError(f"--program-path must point to a .jl file, got: {program_path}")
    image_paths = [Path(p).resolve() for p in args.image_paths]
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")

    prompt = build_prompt(args.audience, program_path)
    print("===== PROMPT BEGIN =====")
    print(prompt)
    print("===== PROMPT END =====")
    if image_paths:
        print("===== IMAGE PATHS =====")
        for image_path in image_paths:
            print(image_path)

    raw_text = call_gemini(prompt, args.model, api_key, image_paths)
    print("===== RESPONSE BEGIN =====")
    print(raw_text)
    print("===== RESPONSE END =====")

    cleaned = strip_code_fences(raw_text)
    parsed = json.loads(cleaned)

    out_path = next_output_path(Path(args.output_dir).resolve(), args.audience, program_path)
    out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
