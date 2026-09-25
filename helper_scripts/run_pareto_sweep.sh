#!/bin/bash
# Usage: run_pareto_sweep.sh <OUTPUT_DIR> <RESNET_TEST_BACC> <MAGE_TEST_BACC> <STAGE>
#   STAGE: fi | audiences | overlay | all
# Reproduces the BloodMNIST plots/false_false tree (runs-per-dim dictionary, LR head, unigram only).
set -u
OUTPUT_DIR=$1; RESNET=$2; MAGE=$3; STAGE=${4:-all}
TRIAL=L16_WD01_asinh_R18_128
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATS=$OUTPUT_DIR/$TRIAL/best_modules_boost_0/asinh/best_modules_program_stats_bestsfalse_false.csv
LOGDIR=${TMPDIR:-/tmp}/magenet-runner-scratch/pareto_logs/$OUTPUT_DIR
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

export JULIA_PYTHONCALL_EXE=$REPO/.CondaPkg/env/bin/python
export JULIA_DEPOT_PATH=/home/$USER/.julia
export JULIAUP_DEPOT_PATH=/home/$USER/.juliaup
export UTCGP_CONSTRAINED=yes UTCGP_MIN_INT=-10000 UTCGP_MAX_INT=10000 UTCGP_MIN_FLOAT=-10000 UTCGP_MAX_FLOAT=10000 UTCGP_SMALL_ARRAY=100 UTCGP_BIG_ARRAY=1000

COMMON=(--trial_id "$TRIAL" --output_dir "$OUTPUT_DIR" --boost_round 0 --val true --test true
        --use_only_bests false --use_really_all false --every_early 1 --every_late 1
        --resnet_test_bacc "$RESNET" --mage_test_bacc "$MAGE" --normalize=true --act=asinh)

run_curve () {  # $1 = label, rest = extra args
  local label=$1; shift
  local t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] START $OUTPUT_DIR $label"
  julia --project -t 16 plots/train_ml_lr_pareto.jl "${COMMON[@]}" "$@" > "$LOGDIR/$label.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $OUTPUT_DIR $label rc=$rc ($(( ($(date +%s)-t0)/60 )) min)"
  return $rc
}

if [[ $STAGE == fi || $STAGE == all ]]; then
  run_curve fi_only
fi

if [[ $STAGE == audiences || $STAGE == all ]]; then
  for aud in cs doctor patient; do
    for red in mean min; do
      run_curve "${aud}_${red}_unigram" --stats_file="$STATS" --audience="$aud" --gram=unigram --reducer="$red" --interpretability_bars=true --show_bar_labels=true --color_bars=false
    done
  done
fi

if [[ $STAGE == overlay || $STAGE == all ]]; then
  for red in mean min; do
    echo "[$(date +%H:%M:%S)] START $OUTPUT_DIR overlay_$red"
    julia --project=plots plots/train_ml_global_pareto_overlay.jl --trial_id="$TRIAL" --output_dir="$OUTPUT_DIR" \
      --use_only_bests=false --use_really_all=false --algorithm=lr --gram=unigram --reducer="$red" \
      --resnet_test_bacc="$RESNET" --mage_test_bacc="$MAGE" > "$LOGDIR/overlay_$red.log" 2>&1
    echo "[$(date +%H:%M:%S)] END   $OUTPUT_DIR overlay_$red rc=$?"
  done
fi
echo "SWEEP DONE $OUTPUT_DIR $STAGE"
