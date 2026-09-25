#!/bin/bash
# Run the RF head for the three all-per-dimension combinations whose SVM and LR rows were
# produced overnight, then merge the rows back together.
#
# src/train_ml.jl rewrites the whole extended-metrics csv with whatever models ran, so the
# svm and lr rows are saved aside first and merged back after the rf run.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S=${TMPDIR:-/tmp}/magenet-runner-scratch
LOG=$S/rf_logs; mkdir -p "$LOG"
cd "$REPO" || exit 1

export JULIA_PYTHONCALL_EXE=$REPO/.CondaPkg/env/bin/python
export JULIA_DEPOT_PATH=/home/$USER/.julia
export JULIAUP_DEPOT_PATH=/home/$USER/.juliaup

for job in "OCTMNIST:L16_WD01_asinh_R18_512:auroc" \
           "OCTMNIST:L16_WD01_asinh_R18_512:ece" \
           "PathMNIST:L16_WD01_asinh_R18_512:macro_f1"; do
  IFS=: read -r OUT TRIAL METRIC <<< "$job"
  CSV=$OUT/$TRIAL/ml_models_boost_0/extended_metrics_${METRIC}_bestsfalse_true.csv
  KEEP=$S/keep_$(basename "$OUT")_${METRIC}.csv
  cp "$CSV" "$KEEP"
  name="$(basename "$OUT")_${METRIC}_rf"
  echo "[$(date +%H:%M:%S)] START $name"
  julia --project -t 8 src/train_ml.jl \
    --trial_id="$TRIAL" --output_dir="$OUT" --boost_round=0 \
    --use_only_bests=false --use_really_all=true \
    --act=asinh --normalize=true --lr_max_iter=3000 \
    --metric_of_interest="$METRIC" --models=rf > "$LOG/$name.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc"
  if [ $rc -eq 0 ]; then
    python3 - "$CSV" "$KEEP" <<'PY'
import csv, sys
rf_path, keep_path = sys.argv[1], sys.argv[2]
rf = list(csv.DictReader(open(rf_path)))
keep = list(csv.DictReader(open(keep_path)))
fields = rf[0].keys() if rf else keep[0].keys()
merged = [r for r in keep if r["model"] != "RF"] + [r for r in rf if r["model"] == "RF"]
order = {"SVC": 0, "RF": 1, "LR": 2}
split_order = {"train": 0, "val": 1, "test": 2}
merged.sort(key=lambda r: (order.get(r["model"], 9), split_order.get(r["split"], 9)))
with open(rf_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(merged)
print("  merged ->", rf_path, "models:", ",".join(sorted({r["model"] for r in merged})))
PY
  fi
done
echo "[$(date +%H:%M:%S)] RF PASS DONE"
