#!/usr/bin/env python3
"""Fill the all-per-dimension cells of the per-dataset metric tables from the result files.

Each table row for a metric has the shape

    Metric (\%)
    & <e2e best> & <e2e avg>
    & RF & SVM & LR        (best per dim)
    & RF & SVM & LR        (runs per dim)
    & RF & SVM & LR        (all per dim)      <- the line this script rewrites
    & <nn> & <nn> & <nn> \\

so the all-per-dim line is the fourth "& ... & ... & ..." after the metric label. Only
lines whose cells are all "--" are replaced, and only when the source file exists, so the
script is safe to run repeatedly as jobs finish.
"""

import csv
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER = next(p for p in REPO.iterdir() if p.is_dir() and p.name.endswith("_Paper"))

# caption fragment -> (output dir, trial id)
TABLES = {
    "Test performance on the OCTMNIST dataset.": ("OCTMNIST", "L16_WD01_asinh_R18_512"),
    "Test performance on the PathMNIST dataset.": ("PathMNIST", "L16_WD01_asinh_R18_512"),
    "Test performance on the OrganCMNIST dataset.": ("ORGANCMNIST", "L16_WD01_asinh_R18_128"),
    "Test performance on the BloodMNIST dataset.": ("BloodMNIST", "L16_WD01_asinh_R18_128"),
    "Test performance on the OrganAMNIST dataset.": ("ORGANAMNIST", "L16_WD01_asinh_R18_128"),
}

# table row label -> metric key in the csv
METRICS = {
    "\\Gls{bacc} (\\%)": "bacc",
    "Macro F1 (\\%)": "macro_f1",
    "ECE (\\%)": "ece",
    "AUROC (\\%)": "auroc",
}


def read_cells(out_dir: str, trial: str, metric: str):
    """RF, SVM, LR test values of the all-per-dimension dictionary, or None."""
    path = REPO / out_dir / trial / "ml_models_boost_0" / f"extended_metrics_{metric}_bestsfalse_true.csv"
    if not path.exists():
        return None
    rows = {r["model"]: r for r in csv.DictReader(path.open()) if r["split"] == "test"}
    if not {"RF", "SVC", "LR"} <= set(rows):
        return None  # a partial file, for instance svc and lr before rf has run
    return [f"{100 * float(rows[m][metric]):.1f}" for m in ("RF", "SVC", "LR")]


def main(apply: bool) -> int:
    text = (PAPER / "appendix.tex").read_text()
    filled = []
    for caption, (out_dir, trial) in TABLES.items():
        if caption not in text:
            continue
        start = text.index(caption)
        end = text.index("\\end{table}", start)
        block = text[start:end]
        new_block = block
        for label, metric in METRICS.items():
            if label not in new_block:
                continue
            seg_start = new_block.index(label)
            seg_end = new_block.index("\\\\", seg_start)
            seg = new_block[seg_start:seg_end]
            groups = re.findall(r"& -- & -- & --", seg)
            if not groups:
                continue  # already filled
            cells = read_cells(out_dir, trial, metric)
            if cells is None:
                continue  # job not finished
            seg_new = seg.replace("& -- & -- & --", "& {} & {} & {}".format(*cells), 1)
            new_block = new_block[:seg_start] + seg_new + new_block[seg_end:]
            filled.append(f"{out_dir} {metric}: {' '.join(cells)}")
        text = text[:start] + new_block + text[end:]

    if not filled:
        print("nothing new to fill")
        return 0
    for f in filled:
        print("filled", f)
    if apply:
        (PAPER / "appendix.tex").write_text(text)
        print(f"\nappendix.tex updated with {len(filled)} row(s)")
    else:
        print("\ndry run, pass --apply to write")
    return len(filled)


if __name__ == "__main__":
    main("--apply" in sys.argv)
