#!/usr/bin/env python3
"""Build unigram lollipop plots comparing audiences, split by library."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
FILENAME_RE = re.compile(r"^(?P<audience>.+)_unigram_(?P<seed>\d+)\.json$")
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["CMU Serif", "Computer Modern Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.facecolor"] = "white"
rcParams["figure.facecolor"] = "white"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the lollipop plotter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("llm_compare", "global_compare"), required=True)
    parser.add_argument("--llm", default=None, help="Required when --mode=llm_compare. Example: chatgpt")
    parser.add_argument("--input_root", default=str(ROOT))
    parser.add_argument("--aggregated_root", default=str(ROOT / "aggregated"))
    parser.add_argument("--output_dir", default=str(ROOT / "plots"))
    parser.add_argument("--top_n", type=int, default=0, help="Optional cap on the number of functions shown. 0 means all.")
    parser.add_argument("--per_page", type=int, default=40, help="Maximum number of functions per PDF page.")
    parser.add_argument("--fig_width", type=float, default=11.0)
    parser.add_argument("--row_height", type=float, default=0.18)
    return parser.parse_args()


def load_payload(path: Path) -> list[dict]:
    """Load one JSON score file and normalize it to a list of library blocks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def library_id(raw: object) -> int:
    """Extract the numeric library id from a string like 'Library 2'."""
    if isinstance(raw, int):
        return raw
    match = re.search(r"(\d+)", str(raw))
    if not match:
        raise ValueError(f"Could not parse library id from {raw!r}")
    return int(match.group(1))


def flatten_scores(path: Path) -> dict[str, float]:
    """Convert one score JSON file into a mapping 'Lk :: unit' -> score."""
    rows: dict[str, float] = {}
    for payload in load_payload(path):
        lib = library_id(payload["library_id"])
        for entry in payload["scores"]:
            key = f"L{lib} :: {entry['unit']}"
            rows[key] = float(entry["score"])
    return rows


def discover_seed_files(input_root: Path, llm: str, audience: str) -> list[Path]:
    """Find all raw unigram seed files for one audience and one LLM."""
    llm_dir = input_root / llm
    if not llm_dir.is_dir():
        raise FileNotFoundError(f"Missing LLM directory: {llm_dir}")
    paths = []
    for path in sorted(llm_dir.glob("*.json")):
        match = FILENAME_RE.match(path.name)
        if match and match.group("audience") == audience:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No unigram seed files found for llm={llm} audience={audience} in {llm_dir}")
    return paths


def discover_llms(aggregated_root: Path) -> list[str]:
    """List available aggregated LLM directories excluding global."""
    llms = [p.name for p in aggregated_root.iterdir() if p.is_dir() and p.name != "global"]
    return sorted(llms)


def build_llm_compare_rows(input_root: Path, llm: str) -> list[dict]:
    """Build one row per function with seed dots separated by audience."""
    audiences = ("cs", "doctor", "patient")
    per_audience = {}
    for audience in audiences:
        paths = discover_seed_files(input_root, llm, audience)
        per_seed = [flatten_scores(path) for path in paths]
        keys = sorted(set().union(*per_seed))
        audience_scores = {}
        for key in keys:
            values = [seed_map[key] for seed_map in per_seed if key in seed_map]
            audience_scores[key] = values
        per_audience[audience] = audience_scores
    keys = sorted(set().union(*per_audience.values()))
    rows = []
    for key in keys:
        audience_payload = {}
        all_vals = []
        for audience, scores in per_audience.items():
            if key not in scores:
                continue
            dots = scores[key]
            audience_payload[audience] = {"dots": dots, "mean": sum(dots) / len(dots)}
            all_vals.extend(dots)
        rows.append({"unit": key, "mean": sum(all_vals) / len(all_vals), "audiences": audience_payload})
    rows.sort(key=lambda row: (-row["mean"], row["unit"]))
    return rows


def build_global_compare_rows(aggregated_root: Path) -> list[dict]:
    """Build one row per function with one line per audience and dots per LLM average."""
    audiences = ("cs", "doctor", "patient")
    llms = discover_llms(aggregated_root)
    per_audience_per_llm = {}
    for audience in audiences:
        llm_payload = {}
        for llm in llms:
            path = aggregated_root / llm / f"{audience}_unigram_avg.json"
            if path.is_file():
                llm_payload[llm] = flatten_scores(path)
        if llm_payload:
            per_audience_per_llm[audience] = llm_payload
    if not per_audience_per_llm:
        raise FileNotFoundError(f"No aggregated unigram files found under {aggregated_root}")
    keys = sorted(
        set().union(
            *[
                set().union(*llm_payload.values())
                for llm_payload in per_audience_per_llm.values()
            ]
        )
    )
    rows = []
    for key in keys:
        audience_payload = {}
        all_vals = []
        for audience, llm_payload in per_audience_per_llm.items():
            dots = [scores[key] for scores in llm_payload.values() if key in scores]
            if not dots:
                continue
            audience_payload[audience] = {"dots": dots, "mean": sum(dots) / len(dots)}
            all_vals.extend(dots)
        rows.append({"unit": key, "mean": sum(all_vals) / len(all_vals), "audiences": audience_payload})
    rows.sort(key=lambda row: (-row["mean"], row["unit"]))
    return rows


def maybe_truncate(rows: list[dict], top_n: int) -> list[dict]:
    """Optionally keep only the first top_n rows."""
    if top_n <= 0:
        return rows
    return rows[:top_n]


def split_rows_by_library(rows: list[dict]) -> dict[int, list[dict]]:
    """Group rows by the leading 'Lk ::' prefix in the unit label."""
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        match = re.match(r"^L(\d+)\s::\s", row["unit"])
        if not match:
            raise ValueError(f"Could not infer library from unit label: {row['unit']}")
        lib = int(match.group(1))
        grouped.setdefault(lib, []).append(row)
    return grouped


def chunk_rows(rows: list[dict], per_page: int) -> list[list[dict]]:
    """Split rows into fixed-size pages."""
    if per_page <= 0:
        return [rows]
    return [rows[i:i + per_page] for i in range(0, len(rows), per_page)]


def padded_labels(rows: list[dict], width: int) -> list[str]:
    """Left-pad labels to a common width so page layouts stay aligned."""
    return [row["unit"].ljust(width) for row in reversed(rows)]


def draw_rows_figure(rows: list[dict], *, title: str, fig_width: float, row_height: float, page_idx: int, n_pages: int, label_width: int):
    """Render one page of the three-lines-per-function lollipop plot."""
    rows = list(rows)
    n = len(rows)
    fig_height = max(6.0, 1.5 + row_height * n * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    block_step = 1.18
    audience_mean_markers = {"cs": "o", "doctor": "P", "patient": "^"}
    audience_colors = {"cs": "#0072B2", "doctor": "#D55E00", "patient": "#009E73"}
    audience_offsets = {"cs": 0.32, "doctor": 0.0, "patient": -0.32}
    audience_labels = {"cs": "CS", "doctor": "Doc", "patient": "Pat"}

    for idx, row in enumerate(rows):
        y_base = (n - 1 - idx) * block_step
        for audience in ("cs", "doctor", "patient"):
            if audience not in row["audiences"]:
                continue
            payload = row["audiences"][audience]
            y = y_base + audience_offsets[audience]
            ax.hlines(y, 1.0, payload["mean"], color=audience_colors[audience], alpha=0.5, linewidth=1.0, zorder=1)
            ax.scatter(
                payload["dots"],
                [y] * len(payload["dots"]),
                s=30,
                facecolors="none",
                edgecolors=audience_colors[audience],
                marker="d",
                linewidths=0.9,
                alpha=0.95,
                zorder=2,
            )
            ax.scatter(
                [payload["mean"]],
                [y],
                s=42,
                color=audience_colors[audience],
                marker=audience_mean_markers[audience],
                linewidths=1.2,
                zorder=3,
            )
            ax.text(0.92, y, audience_labels[audience], va="center", ha="right", fontsize=7, color=audience_colors[audience])

    ytick_positions = [i * block_step for i in range(n)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(padded_labels(rows, label_width), fontsize=9, fontweight="bold", fontfamily="monospace")
    ax.set_xlim(0.6, 10.2)
    ax.set_xlabel("Interpretability score")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.0)
    ax.set_ylim(-0.55, (n - 1) * block_step + 0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", labelsize=10, width=0.8, length=4)
    ax.tick_params(axis="y", width=0.0, length=0)
    fig.tight_layout()
    return fig


def plot_rows(rows: list[dict], *, title: str, output_path: Path, fig_width: float, row_height: float, per_page: int) -> None:
    """Render and save the three-lines-per-function lollipop plot as a multi-page PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages = chunk_rows(rows, per_page)
    label_width = max(len(row["unit"]) for row in rows)
    with PdfPages(output_path) as pdf:
        for page_idx, page_rows in enumerate(pages, start=1):
            fig = draw_rows_figure(
                page_rows,
                title=title,
                fig_width=fig_width,
                row_height=row_height,
                page_idx=page_idx,
                n_pages=len(pages),
                label_width=label_width,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def output_path(output_dir: Path, mode: str, llm: str | None, lib: int) -> Path:
    """Choose a predictable PDF path for one lollipop plot."""
    if mode == "llm_compare":
        assert llm is not None
        return output_dir / f"{llm}_audiences_unigram_lollipop_lib{lib}.pdf"
    return output_dir / f"global_audiences_unigram_lollipop_lib{lib}.pdf"


def main() -> None:
    """Run the requested lollipop plot job."""
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    aggregated_root = Path(args.aggregated_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.mode == "llm_compare":
        if not args.llm:
            raise ValueError("--llm is required when --mode=llm_compare")
        rows = build_llm_compare_rows(input_root, args.llm)
        rows = maybe_truncate(rows, args.top_n)
        for lib, lib_rows in sorted(split_rows_by_library(rows).items()):
            plot_rows(
                lib_rows,
                title=f"Unigram scores by audience, {args.llm}, Library {lib}",
                output_path=output_path(output_dir, args.mode, args.llm, lib),
                fig_width=args.fig_width,
                row_height=args.row_height,
                per_page=args.per_page,
            )
    else:
        rows = build_global_compare_rows(aggregated_root)
        rows = maybe_truncate(rows, args.top_n)
        for lib, lib_rows in sorted(split_rows_by_library(rows).items()):
            plot_rows(
                lib_rows,
                title=f"Unigram scores by audience, global averages, Library {lib}",
                output_path=output_path(output_dir, args.mode, None, lib),
                fig_width=args.fig_width,
                row_height=args.row_height,
                per_page=args.per_page,
            )


if __name__ == "__main__":
    main()
