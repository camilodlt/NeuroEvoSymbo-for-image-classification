#!/bin/bash
# Two remaining RF heads, run sequentially because RandomForestClassifier is hardcoded to
# n_jobs=14 regardless of the julia thread count, so two at once would oversubscribe.
#
#  1. OCT  all-per-dim, balanced-accuracy scored -> fills Table 1 All/dim RF and the OCT ECE row
#  2. Path all-per-dim, macro F1 scored          -> fills the Path macro F1 row
#
# src/train_ml.jl rewrites the whole extended-metrics csv with only the models that ran, so
# every target file is saved aside first and the rows merged back afterwards.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S=${TMPDIR:-/tmp}/magenet-runner-scratch/rf_tonight
mkdir -p "$S/logs" "$S/backup"
cd "$REPO" || exit 1

export JULIA_PYTHONCALL_EXE=$REPO/.CondaPkg/env/bin/python
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$HOME/.julia}
export JULIAUP_DEPOT_PATH=${JULIAUP_DEPOT_PATH:-$HOME/.juliaup}
export UTCGP_CONSTRAINED=yes UTCGP_MIN_INT=-10000 UTCGP_MAX_INT=10000
export UTCGP_MIN_FLOAT=-10000 UTCGP_MAX_FLOAT=10000
export UTCGP_SMALL_ARRAY=100 UTCGP_BIG_ARRAY=1000

merge_rows () {   # $1 = file that now holds only RF, $2 = file holding the other heads
  python3 - "$1" "$2" <<'PY'
import csv, sys
rf_path, donor_path = sys.argv[1], sys.argv[2]
rf   = list(csv.DictReader(open(rf_path)))
other= [r for r in csv.DictReader(open(donor_path)) if r["model"] != "RF"]
fields = list(rf[0].keys()) if rf else list(other[0].keys())
merged = other + [r for r in rf if r["model"] == "RF"]
order  = {"SVC":0,"RF":1,"LR":2}; sp = {"train":0,"val":1,"test":2}
merged.sort(key=lambda r:(order.get(r["model"],9), sp.get(r["split"],9)))
with open(rf_path,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(merged)
print("   merged ->", rf_path, "models:", ",".join(sorted({r["model"] for r in merged})))
PY
}

# ---------------- 1. OCT all-per-dim, bacc ----------------
OCT=$REPO/OCTMNIST/L16_WD01_asinh_R18_512/ml_models_boost_0
cp "$OCT/extended_metrics_ece_bestsfalse_true.csv" "$S/backup/oct_ece_allperdim.csv"
echo "[$(date +%H:%M:%S)] START oct_rf_allperdim_bacc"
t0=$(date +%s)
julia --project -t 2 src/train_ml.jl \
  --trial_id=L16_WD01_asinh_R18_512 --output_dir=OCTMNIST --boost_round=0 \
  --use_only_bests=false --use_really_all=true \
  --act=asinh --normalize=true --lr_max_iter=3000 \
  --metric_of_interest=bacc --models=rf > "$S/logs/oct_rf.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] END   oct_rf_allperdim_bacc rc=$rc ($(( ($(date +%s)-t0)/60 )) min)"
if [ $rc -eq 0 ]; then
  BACC=$OCT/extended_metrics_bacc_bestsfalse_true.csv
  awk -F, 'NR>1 && $2=="test" && $1=="RF"{printf "   OCT all-per-dim RF test bacc = %.2f   ece = %.2f\n",100*$6,100*$12}' "$BACC"
  # complete the bacc file with the balanced-accuracy-scored SVC and LR already on disk
  merge_rows "$BACC" "$S/backup/oct_ece_allperdim.csv"
  # and give the ece file its RF rows, same scorer so the same fitted models
  cp "$S/backup/oct_ece_allperdim.csv" "$OCT/extended_metrics_ece_bestsfalse_true.csv"
  python3 - "$OCT/extended_metrics_ece_bestsfalse_true.csv" "$BACC" <<'PY'
import csv, sys
dst, src = sys.argv[1], sys.argv[2]
keep = [r for r in csv.DictReader(open(dst)) if r["model"] != "RF"]
rf   = [r for r in csv.DictReader(open(src)) if r["model"] == "RF"]
for r in rf: r["metric_of_interest"] = "ece"; r["primary"] = r["ece"]
merged = keep + rf
order={"SVC":0,"RF":1,"LR":2}; sp={"train":0,"val":1,"test":2}
merged.sort(key=lambda r:(order.get(r["model"],9), sp.get(r["split"],9)))
with open(dst,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(merged[0].keys())); w.writeheader(); w.writerows(merged)
print("   merged ->", dst, "models:", ",".join(sorted({r["model"] for r in merged})))
PY
fi

# ---------------- 2. Path all-per-dim, macro F1 ----------------
PATHD=$REPO/PathMNIST/L16_WD01_asinh_R18_512/ml_models_boost_0
PF=$PATHD/extended_metrics_macro_f1_bestsfalse_true.csv
cp "$PF" "$S/backup/path_macrof1_allperdim.csv"
echo "[$(date +%H:%M:%S)] START path_rf_allperdim_macro_f1"
t0=$(date +%s)
julia --project -t 2 src/train_ml.jl \
  --trial_id=L16_WD01_asinh_R18_512 --output_dir=PathMNIST --boost_round=0 \
  --use_only_bests=false --use_really_all=true \
  --act=asinh --normalize=true --lr_max_iter=3000 \
  --metric_of_interest=macro_f1 --models=rf > "$S/logs/path_rf.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] END   path_rf_allperdim_macro_f1 rc=$rc ($(( ($(date +%s)-t0)/60 )) min)"
if [ $rc -eq 0 ]; then
  awk -F, 'NR>1 && $2=="test" && $1=="RF"{printf "   Path all-per-dim RF test macro_f1 = %.2f\n",100*$7}' "$PF"
  merge_rows "$PF" "$S/backup/path_macrof1_allperdim.csv"
fi
echo "[$(date +%H:%M:%S)] RF TONIGHT DONE"
