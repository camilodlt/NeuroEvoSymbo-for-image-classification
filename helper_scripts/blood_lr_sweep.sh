#!/bin/bash
# Regenerate every BloodMNIST LR interpretability-performance curve.
#
# The figures currently in the paper were produced on 30 April and 1 May, before the vertical
# bar labels were given overlap avoidance in plots/ml_utils.jl (place_bar_labels!, 17 August),
# so they are redrawn here with the current code.
#
#   1 feature-importance-only reference
#   3 audiences x {unigram, bigram} x {mean, min} = 12 audience curves
#
# Baselines are the corrected values, best ResNet 99.1 and validation-selected end-to-end 67.3.
#
# Each run averages about 0.6 of a core and holds roughly 2.3 GB, measured, so they are run
# in parallel batches rather than sequentially. BATCH keeps the peak well inside 16 cores.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S=${TMPDIR:-/tmp}/magenet-runner-scratch/blood_lr
LOG=$S/logs; mkdir -p "$LOG"
cd "$REPO" || exit 1

OUT=BloodMNIST
TRIAL=L16_WD01_asinh_R18_128
STATS=$OUT/$TRIAL/best_modules_boost_0/asinh/best_modules_program_stats_bestsfalse_false.csv
RESNET=0.991
MAGE=0.673
BATCH=${BATCH:-7}
THREADS=${THREADS:-2}

export JULIA_PYTHONCALL_EXE=$REPO/.CondaPkg/env/bin/python
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$HOME/.julia}
export JULIAUP_DEPOT_PATH=${JULIAUP_DEPOT_PATH:-$HOME/.juliaup}
export UTCGP_CONSTRAINED=yes UTCGP_MIN_INT=-10000 UTCGP_MAX_INT=10000
export UTCGP_MIN_FLOAT=-10000 UTCGP_MAX_FLOAT=10000
export UTCGP_SMALL_ARRAY=100 UTCGP_BIG_ARRAY=1000

COMMON=(--trial_id "$TRIAL" --output_dir "$OUT" --boost_round 0 --val true --test true
        --use_only_bests false --use_really_all false --every_early 1 --every_late 1
        --resnet_test_bacc "$RESNET" --mage_test_bacc "$MAGE" --normalize=true --act=asinh)

launch () {   # $1 = label, rest = extra args
  local label=$1; shift
  ( t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] START $label"
    julia --project -t "$THREADS" plots/train_ml_lr_pareto.jl "${COMMON[@]}" "$@" \
      > "$LOG/$label.log" 2>&1
    echo "[$(date +%H:%M:%S)] END   $label rc=$? ($(( ($(date +%s)-t0)/60 )) min)"
  ) &
}

echo "[$(date +%H:%M:%S)] BLOOD LR SWEEP START, 13 runs, batches of $BATCH, -t $THREADS"

# build the job list
JOBS=("fi_only|")
for aud in cs doctor patient; do
  for gram in unigram bigram; do
    for red in mean min; do
      JOBS+=("${aud}_${red}_${gram}|--stats_file=$STATS --audience=$aud --gram=$gram --reducer=$red --interpretability_bars=true --show_bar_labels=true --color_bars=false")
    done
  done
done

i=0
for job in "${JOBS[@]}"; do
  label=${job%%|*}; args=${job#*|}
  # shellcheck disable=SC2086
  launch "$label" $args
  i=$((i+1))
  if [ $((i % BATCH)) -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] --- waiting for batch to finish ($i of ${#JOBS[@]}) ---"
    wait
  fi
done
wait
echo "[$(date +%H:%M:%S)] BLOOD LR SWEEP DONE"
echo "--- figures produced in the last hour ---"
find "$REPO/$OUT/$TRIAL/plots" -name "*pareto_plot_lr*" -newermt "-1 hour" -printf "%TH:%TM %f\n" | sort
