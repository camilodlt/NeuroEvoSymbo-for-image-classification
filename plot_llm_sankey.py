#!/usr/bin/env python3
"""Build a Sankey diagram showing score flow from CS to doctor to patient."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
AUDIENCES = ("cs", "doctor", "patient")
AUDIENCE_COLORS = {
    "cs": "#0072B2",
    "doctor": "#D55E00",
    "patient": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregated_root", default=str(ROOT / "aggregated" / "global"))
    parser.add_argument("--gram", choices=("unigram", "bigram"), required=True)
    parser.add_argument("--output_dir", default=str(ROOT / "plots"))
    return parser.parse_args()


def parse_library_id(raw_value) -> int:
    if isinstance(raw_value, int):
        return raw_value
    s = str(raw_value)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse library_id from {raw_value!r}")
    return int(digits)


def load_score_map(path: Path) -> dict[tuple[int, str], int]:
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    payloads = parsed if isinstance(parsed, list) else [parsed]
    vals: dict[tuple[int, str], int] = {}
    for payload in payloads:
        library_id = parse_library_id(payload["library_id"])
        for entry in payload["scores"]:
            unit = str(entry["unit"])
            vals[(library_id, unit)] = int(round(float(entry["score"])))
    return vals


def build_triplets(aggregated_root: Path, gram: str) -> list[tuple[int, int, int]]:
    score_maps = {}
    for audience in AUDIENCES:
        path = aggregated_root / f"{audience}_{gram}_avg.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing aggregated score file: {path}")
        score_maps[audience] = load_score_map(path)

    lengths = {audience: len(vals) for audience, vals in score_maps.items()}
    shared_keys = set.intersection(*(set(vals.keys()) for vals in score_maps.values()))
    if not shared_keys:
        raise ValueError(f"No shared keys across audiences for gram={gram}. Sizes: {lengths}")
    dropped = {audience: lengths[audience] - len(shared_keys) for audience in AUDIENCES}
    print(f"[info] Sankey alignment gram={gram} total_sizes={lengths} shared={len(shared_keys)} dropped={dropped}")
    ordered_keys = sorted(shared_keys)
    return [(score_maps["cs"][k], score_maps["doctor"][k], score_maps["patient"][k]) for k in ordered_keys]


def node_name(audience: str, score: int) -> str:
    return f"{audience}:{score}"


def build_sankey(triplets: list[tuple[int, int, int]], gram: str) -> go.Figure:
    labels = []
    colors = []
    node_index = {}
    score_order = list(range(10, 0, -1))
    for audience in AUDIENCES:
        for score in score_order:
            name = node_name(audience, score)
            node_index[name] = len(labels)
            labels.append(f"{audience.upper()} {score}")
            colors.append(AUDIENCE_COLORS[audience])

    cs_doctor = Counter((cs, doctor) for cs, doctor, _patient in triplets)
    doctor_patient = Counter((doctor, patient) for _cs, doctor, patient in triplets)

    sources = []
    targets = []
    values = []
    link_colors = []

    for (cs, doctor), count in sorted(cs_doctor.items()):
        sources.append(node_index[node_name("cs", cs)])
        targets.append(node_index[node_name("doctor", doctor)])
        values.append(count)
        link_colors.append("rgba(0,114,178,0.35)")

    for (doctor, patient), count in sorted(doctor_patient.items()):
        sources.append(node_index[node_name("doctor", doctor)])
        targets.append(node_index[node_name("patient", patient)])
        values.append(count)
        link_colors.append("rgba(213,94,0,0.35)")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=10,
                    thickness=16,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Global {gram.capitalize()} score flow: CS → Doctor → Patient",
        font=dict(family="CMU Serif, Times New Roman, serif", size=16),
        width=1400,
        height=900,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def main() -> None:
    args = parse_args()
    aggregated_root = Path(args.aggregated_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    triplets = build_triplets(aggregated_root, args.gram)
    fig = build_sankey(triplets, args.gram)

    out_html = output_dir / f"sankey_{args.gram}.html"
    out_pdf = output_dir / f"sankey_{args.gram}.pdf"
    fig.write_html(out_html, include_plotlyjs="cdn")
    fig.write_image(out_pdf)
    print(out_html)
    print(out_pdf)


if __name__ == "__main__":
    main()
