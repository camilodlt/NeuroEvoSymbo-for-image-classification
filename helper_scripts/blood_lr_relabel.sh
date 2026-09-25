#!/bin/bash
# Redraw the BloodMNIST audience curves after the bar-label gap fix in plots/ml_utils.jl.
#
# place_bar_labels! measured the collision threshold against the span of the bars themselves,
# so a tightly clustered set shrank the threshold with itself and every label landed on one
# row. It now measures against the x axis, passed as axis_span from the call site.
#
# patient_min_bigram was already redrawn as the validation case, and fi_only has no bars,
# so eleven figures remain.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S=${TMPDIR:-/tmp}/magenet-runner-scratch/blood_lr
LOG=$S/logs; mkdir -p "$LOG"
cd "$REPO" || exit 1

OUT=BloodMNIST
TRIAL=L16_WD01_asinh_R18_128
STATS=$OUT/$TRIAL/best_modules_boost_0/asinh/best_modules_program_stats_bestsfalse_false.csv
BATCH=${BATCH:-6}

export JULIA_PYTHONCALL_EXE=$REPO/.CondaPkg/env/bin/python
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$HOME/.julia}
export JULIAUP_DEPOT_PATH=${JULIAUP_DEPOT_PATH:-$HOME/.juliaup}
export UTCGP_CONSTRAINED=yes UTCGP_MIN_INT=-10000 UTCGP_MAX_INT=10000
export UTCGP_MIN_FLOAT=-10000 UTCGP_MAX_FLOAT=10000
export UTCGP_SMALL_ARRAY=100 UTCGP_BIG_ARRAY=1000

COMMON=(--trial_id "$TRIAL" --output_dir "$OUT" --boost_round 0 --val true --test true
        --use_only_bests false --use_really_all false --every_early 1 --every_late 1
        --resnet_test_bacc 0.991 --mage_test_bacc 0.673 --normalize=true --act=asinh)

echo "[$(date +%H:%M:%S)] RELABEL SWEEP START, 11 figures, batches of $BATCH"
i=0
for aud in cs doctor patient; do
  for gram in unigram bigram; do
    for red in mean min; do
      [ "$aud$gram$red" = "patientbigrammin" ] && continue    # already redrawn
      label="${aud}_${red}_${gram}"
      ( t0=$(date +%s); echo "[$(date +%H:%M:%S)] START $label"
        julia --project -t 2 plots/train_ml_lr_pareto.jl "${COMMON[@]}" \
          --stats_file="$STATS" --audience="$aud" --gram="$gram" --reducer="$red" \
          --interpretability_bars=true --show_bar_labels=true --color_bars=false \
          > "$LOG/${label}_fixed.log" 2>&1
        echo "[$(date +%H:%M:%S)] END   $label rc=$? ($(( ($(date +%s)-t0)/60 )) min)" ) &
      i=$((i+1))
      if [ $((i % BATCH)) -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] --- waiting for batch ($i of 11) ---"; wait
      fi
    done
  done
done
wait
echo "[$(date +%H:%M:%S)] RELABEL SWEEP DONE"
