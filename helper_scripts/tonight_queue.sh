#!/bin/bash
# Tonight's queue, started automatically once the OrganA pareto sweep is done.
#   1. OrganC train_ml with per-metric head selection (3 metrics x 3 dictionaries)
#   2. OrganC frontier sweep (fi + 6 audience curves + 2 overlays)
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S=${TMPDIR:-/tmp}/magenet-runner-scratch
LOGDIR=$S/tonight_logs; mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

export JULIA_PYTHONCALL_EXE=$REPO/.CondaPkg/env/bin/python
export JULIA_DEPOT_PATH=/home/$USER/.julia
export JULIAUP_DEPOT_PATH=/home/$USER/.juliaup
export UTCGP_CONSTRAINED=yes UTCGP_MIN_INT=-10000 UTCGP_MAX_INT=10000 UTCGP_MIN_FLOAT=-10000 UTCGP_MAX_FLOAT=10000 UTCGP_SMALL_ARRAY=100 UTCGP_BIG_ARRAY=1000

# wait for the OrganA sweep to finish before taking the CPU
while pgrep -f "train_ml_lr_pareto.jl|train_ml_global_pareto_overlay.jl" > /dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] OrganA sweep finished, starting tonight's queue"

# ---- 1. OrganC per-metric head selection ----
for m in macro_f1 auroc ece; do
  for cfg in "true:false:beststrue_false" "false:false:bestsfalse_false" "false:true:bestsfalse_true"; do
    IFS=: read -r bests really tag <<< "$cfg"
    name="trainml_${m}_${tag}"
    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] START $name"
    julia --project -t 16 src/train_ml.jl \
      --trial_id="L16_WD01_asinh_R18_128" \
      --output_dir="ORGANCMNIST" \
      --boost_round=0 \
      --use_only_bests=$bests \
      --use_really_all=$really \
      --act=asinh \
      --normalize=true \
      --lr_max_iter=3000 \
      --metric_of_interest=$m > "$LOGDIR/$name.log" 2>&1
    echo "[$(date +%H:%M:%S)] END   $name rc=$? ($(( ($(date +%s)-t0)/60 )) min)"
  done
done
echo "[$(date +%H:%M:%S)] TRAIN_ML DONE"

# ---- 2. OrganC frontier sweep ----
# resnet_test_bacc 0.944 (ResNeXt50), mage_test_bacc 0.374 (validation-selected E2E run)
"$S/run_pareto_sweep.sh" ORGANCMNIST 0.944 0.374 all
echo "[$(date +%H:%M:%S)] TONIGHT QUEUE DONE"
