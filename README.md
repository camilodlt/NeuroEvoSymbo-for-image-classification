## Data layout

The repository expects MedMNIST datasets stored as `.jld2` files under `datasets_pickle/`. For each dataset, three files are used: `*_train.jld2`, `*_val.jld2`, and `*_test.jld2`. The experiments in this submission use `BloodMNIST`, `PathMNIST`, `OrganAMNIST`, `OrganCMNIST`, and `OCTMNIST`.

The `.jld2` dataset files are not included in this submission repository because they are too large. To reproduce the experiments, the reader must obtain the corresponding datasets separately and place them under `datasets_pickle/` using the expected file names.

## Results reported in the paper

Every number in the paper's results tables is read from a file in this repository. The
dataset folders carry the result files. The manuscript sources are distributed separately,
so checking the tables against the LaTeX is an author-side validation step.

| Paper table | Content | Files |
| --- | --- | --- |
| Table 1 | Test BACC across datasets | all four sources below |
| Table 4 | Bottlenecked teacher against the baselines | `nn_surrogates_boost_*/metrics.txt`, baseline json |
| Table 7 | Cross-dataset summary of macro F1, ECE and AUROC | same as Tables 8 to 12 |
| Tables 8 to 12 | Per-dataset BACC, macro F1, ECE and AUROC | `ml_models_boost_0/extended_metrics_*.csv` |
| Tables 13 to 17 | Per-class sensitivity and specificity | the same csv files plus `per_class_test.csv` |

The four sources, under `<DATASET>/`:

| Source | Path | Holds |
| --- | --- | --- |
| MAGE end-to-end | `MAGE_ALONE_20H/mage_imgcls/metrics_per_individual_<split>_alltrue.csv` | one row per run, all metrics. The validation-selected run is the one with the highest validation BACC, and reported spreads are sample standard deviations over the 16 runs. `metrics_summary_*.csv` holds the same aggregated, with a population standard deviation, which is why the paper recomputes from the per-run file. |
| Distillation | `<TRIAL>/ml_models_boost_0/extended_metrics_<metric>_<dict>.csv` | train, val and test rows for the RF, SVM and LR heads, with per-class sensitivity and specificity as semicolon-separated vectors |
| Neural baselines | `NN_baselines/<backbone>_<name>/extended_metrics_best_val_bacc.json` and `per_class_test.csv` | test BACC, macro F1, AUROC and ECE of the checkpoint with the highest validation BACC |
| Teacher | `<TRIAL>/nn_surrogates_boost_*/metrics.txt` | three values, train, validation and test BACC of the bottlenecked teacher |

`<dict>` names the program dictionary, `beststrue_false` for Best per dimension,
`bestsfalse_false` for Runs per dimension and `bestsfalse_true` for All per dimension.

Each MAGE run directory, whether a distillation search under `<TRIAL>/1_<coordinate>/` or an
end-to-end search under `MAGE_ALONE_20H/mage_imgcls/`, ships two files. `metrics_<run>.json`
is the per-generation trace, one JSON object per line with the loss, the fitness, the
iteration, the split and the elapsed time. `checkpoint_0.pickle` is the serialised genome of
the validation-best individual, rewritten on every validation improvement rather than being
the initial population, and it is the file the readers load to recover a run's selected
program. The intermediate `checkpoint_<generation>.pickle` files are not included, since each
search writes on the order of two thousand of them.

Two conventions are worth stating, because they are easy to misread from the filenames.
The ECE rows are read from the `bacc`-scored file, since scikit-learn has no ECE scorer and
`--metric_of_interest=ece` selects hyperparameters with balanced accuracy, so the ECE row
describes the same fitted heads as the BACC row. And `src/train_ml.jl` rewrites the whole
csv with only the models that ran, so a run with `--models=rf` replaces the SVM and LR rows
unless they are saved aside and merged back.

## Environment

Julia dependencies are specified in `Project.toml`.

The Julia-based experiments in this repository were run with:

```bash
export JULIA_DEPOT_PATH=<path-to-julia-depot>
export JULIAUP_DEPOT_PATH=<path-to-juliaup-depot>
export JULIA_PYTHONCALL_EXE=./.CondaPkg/env/bin/python
export UTCGP_CONSTRAINED=yes
export UTCGP_MIN_INT=-10000
export UTCGP_MAX_INT=10000
export UTCGP_MIN_FLOAT=-10000
export UTCGP_MAX_FLOAT=10000
export UTCGP_SMALL_ARRAY=100
export UTCGP_BIG_ARRAY=1000
```

When running Julia scripts, use the repository project:

```bash
julia --project ...
```

## Neural-network baselines

The CNN baselines reported in the paper are trained from the dataset-specific Jupyter notebooks in:

- `BloodMNIST/NN_baselines/`
- `PathMNIST/NN_baselines/`
- `ORGANAMNIST/NN_baselines/`
- `OCTMNIST/NN_baselines/`

These notebooks use shared helper code from `python_utils/`, in particular:

- `python_utils/medmnist_resnet_baseline.py`
- `python_utils/jld2_python.py`

The main Python dependencies for these notebooks are PyTorch, torchvision, numpy, pandas, scikit-learn, juliacall, and Jupyter.

## MAGE-alone baseline

The MAGE-alone baseline is obtained by running `src/train_mage_imgcls.jl` directly. In the paper, this experiment was repeated `16` times per dataset with varying seeds.

The following command illustrates the configuration used in the paper:

```bash
julia --project --threads=20 src/train_mage_imgcls.jl \
  --seed=1 \
  --data_location=datasets_pickle/BloodMNIST_64_train.jld2 \
  --val_data_location=datasets_pickle/BloodMNIST_64_val.jld2 \
  --output_dir=BloodMNIST/MAGE_ALONE_20H \
  --trial_id=MAGE_ALONE_20H \
  --gens=100000000 \
  --n_nodes=100 \
  --n_new=114 \
  --n_elite=16 \
  --tour_size=7 \
  --n_samples=72 \
  --err_w=0.8 \
  --time_w=100.0 \
  --time=20.0 \
  --act=identity \
  --use_ski=false \
  --use_new_number_extensions=true \
  --use_new_intensityimg_extensions=true \
  --use_new_binaryimg_extensions=true \
  --use_new_segmentimg_extensions=false \
  --use_imagegraph_bundle=false
```

## Reading MAGE-alone results

After training, the saved MAGE-alone runs can be read and evaluated on the train, validation, and test sets with `src/read_best_magecls_per_trial.jl`.

```bash
julia --project -t 16 src/read_best_magecls_per_trial.jl \
  --data_location="datasets_pickle/PathMNIST_64_train.jld2" \
  --val_data_location="datasets_pickle/PathMNIST_64_val.jld2" \
  --test_data_location="datasets_pickle/PathMNIST_64_test.jld2" \
  --trial_id="MAGE_ALONE_20H" \
  --output_dir="PathMNIST" \
  --use_ski=false \
  --all=true \
  --act="identity" \
  --use_new_number_extensions=true \
  --use_new_intensityimg_extensions=true \
  --use_new_binaryimg_extensions=true \
  --use_new_segmentimg_extensions=false \
  --use_imagegraph_bundle=false
```

With `--all=true` the reader evaluates every run, which is what the mean and standard
deviation across runs are computed from. With `--all=false` it evaluates only the
validation-selected run, which is much faster and enough when only that program is
needed. The reader loads `checkpoint_0.pickle`, which stores the validation-best
individual found in that evolutionary run.

Every output carries the setting in its name, `metrics_summary_alltrue.csv` against
`metrics_summary_allfalse.csv` and likewise for the per-individual, per-class and
ensemble files, so a single-individual read never overwrites the statistics of a full
read.

## Teacher training

The distillation teacher is trained with `src/distill/train_resnet18_latent_global.jl`.

```bash
julia --project -t 16 src/distill/train_resnet18_latent_global.jl \
  --latent_dim=16 \
  --lip=false \
  --val_bs=512 \
  --workers=true \
  --pretrained="resnet18" \
  --epochs=100 \
  --train_bs=512 \
  --last_layer_type=vsmall \
  --lr=0.0001 \
  --wd=0.1 \
  --schedule=false \
  --mix_prob=0. \
  --data_location=datasets_pickle/PathMNIST_64_train.jld2 \
  --val_data_location=datasets_pickle/PathMNIST_64_val.jld2 \
  --test_data_location=datasets_pickle/PathMNIST_64_test.jld2 \
  --output_dir=PathMNIST \
  --trial_id=L16_WD01_asinh_R18_512 \
  --optim=adamw \
  --tail_activation=asinh \
  --torchvision_weights=default \
  --resize_to=224 \
  --distill_phase=all
```

In the paper, `train_bs=512` was used for the larger datasets and `train_bs=128` for the smaller datasets.

## Training MAGE students for teacher coordinates

The MAGE students fitted to teacher coordinates are trained with `src/fit_surrogate.jl`. In the paper, this procedure was run separately for each teacher coordinate.

The command below is the direct Julia equivalent of the launcher used in the experiments:

```bash
julia --project -t 20 src/fit_surrogate.jl \
  --seed=1 \
  --data_location=PathMNIST/L16_WD01_asinh_R18_512/nn_surrogates_boost_0/surrogate_1_1_train.jld2 \
  --val_data_location=PathMNIST/L16_WD01_asinh_R18_512/nn_surrogates_boost_0/surrogate_1_1_val.jld2 \
  --output_dir=PathMNIST \
  --trial_id=L16_WD01_asinh_R18_512 \
  --gens=2000000 \
  --mutation_rate=1 \
  --n_nodes=30 \
  --time=20 \
  --time_w=1000 \
  --n_new=128 \
  --n_elite=16 \
  --tour_size=7 \
  --n_samples=64 \
  --loss_type=corr \
  --use_ski=false \
  --act=sigmoid \
  --use_new_number_extensions=true \
  --use_new_intensityimg_extensions=true \
  --use_new_binaryimg_extensions=true \
  --use_new_segmentimg_extensions=false \
  --use_imagegraph_bundle=false \
  --boost_K=1_1
```

The value of `n_samples` is dataset dependent; see the paper for the exact per-dataset configuration.

## Reading distiller results

Once the distillers are trained, their results can be read with `src/read_best_models_per_trial.jl`.

In the paper:

- *Best per dim* corresponds to `--use_only_bests=true --use_really_all=false`
- *Runs per dim* corresponds to `--use_only_bests=false --use_really_all=false`
- *All per dim* corresponds to `--use_only_bests=false --use_really_all=true`

For example:

```bash
julia --project -t 16 src/read_best_models_per_trial.jl \
  --data_location datasets_pickle/OCTMNIST_64_train.jld2 \
  --val_data_location datasets_pickle/OCTMNIST_64_val.jld2 \
  --test_data_location datasets_pickle/OCTMNIST_64_test.jld2 \
  --trial_id L16_WD01_asinh_R18_512 \
  --output_dir OCTMNIST \
  --boost_round 0 \
  --act_family regression \
  --use_only_bests false \
  --use_really_all true \
  --multiple_of 1000 \
  --metric_of_interest loss \
  --use_ski false \
  --use_new_number_extensions true \
  --use_new_intensityimg_extensions true \
  --use_new_binaryimg_extensions true \
  --use_new_segmentimg_extensions false \
  --use_imagegraph_bundle false \
  --single_grade_files aggregated/global/cs_unigram_avg.json,aggregated/global/doctor_unigram_avg.json,aggregated/global/patient_unigram_avg.json
```

The `multiple_of` argument controls every how many generations the best individual of the generation is loaded when `--use_really_all=true`.

## Training the ML models

The downstream ML models are trained with `src/train_ml.jl`.

```bash
julia --project -t 16 src/train_ml.jl \
  --trial_id="L16_WD01_asinh_R18_512" \
  --output_dir="OCTMNIST" \
  --boost_round=0 \
  --use_only_bests=false \
  --use_really_all=true \
  --act=asinh \
  --normalize=true \
  --lr_max_iter=3000
```

The dataset and experiment variant are controlled by `output_dir` and `trial_id`.

## Audience-aware Pareto plots

Audience-aware tradeoff plots can be generated for unigram or bigram interpretability with the Pareto plotting scripts in `plots/`.

For example:

```bash
julia --project plots/train_ml_rf_pareto.jl \
  --trial_id L16_WD01_asinh_R18_128 \
  --output_dir BloodMNIST \
  --boost_round 0 \
  --val true \
  --test true \
  --use_only_bests true \
  --use_really_all false \
  --every_early 1 \
  --every_late 1 \
  --resnet_test_bacc 0.99 \
  --mage_test_bacc 0.66 \
  --normalize=true \
  --act=asinh \
  --stats_file=BloodMNIST/L16_WD01_asinh_R18_128/best_modules_boost_0/asinh/best_modules_program_stats_beststrue_false.csv \
  --audience=cs \
  --gram=unigram \
  --reducer=mean \
  --interpretability_bars=true
```

If `stats_file`, `audience`, `gram`, and `reducer` are omitted, the plot reduces to the feature-importance-only tradeoff.

In the paper, these plots are generated for all audiences, both grams, and both reducers (`min` and `mean`).

## Overlay Pareto plots

Once the audience-aware Pareto plots have been generated for all audiences, grams, and reducers, they can be combined into a single overlay plot with `plots/train_ml_global_pareto_overlay.jl`.

```bash
julia --project=plots plots/train_ml_global_pareto_overlay.jl \
  --trial_id=L16_WD01_asinh_R18_128 \
  --output_dir=BloodMNIST \
  --use_only_bests=false \
  --use_really_all=false \
  --algorithm=lr \
  --gram=unigram \
  --reducer=mean \
  --resnet_test_bacc=0.99 \
  --mage_test_bacc=0.66
```

## LLM scoring

The unigram and bigram interpretability scores are generated with the Python caller scripts:

- `run_gemini_prompt.py`
- `run_anthropic_prompt.py`
- `run_openai_prompt.py`

Each script expects the corresponding API key to be available in the environment.

Examples:

```bash
python3 run_gemini_prompt.py cs unigram --model gemini-3.1-pro-preview
python3 run_gemini_prompt.py cs bigram --model gemini-3.1-pro-preview

python3 run_anthropic_prompt.py cs unigram --model claude-sonnet-4-6 --max-tokens=64000
python3 run_anthropic_prompt.py cs bigram --model claude-sonnet-4-6 --max-tokens=64000

python3 run_openai_prompt.py cs unigram --model gpt-5
python3 run_openai_prompt.py cs bigram --model gpt-5
```

These commands are run separately for each audience (`cs`, `doctor`, `patient`). The unigram callers write one JSON file per run. The bigram callers split the request internally by library and merge the result back into a single JSON file per run.

## Whole-program LLM interpretation

Whole-program interpretations can be generated with `run_gemini_whole_program.py`, which takes a full exported program file through `--program-path`.

For example:

```bash
python3 run_gemini_whole_program.py \
  cs \
  --program-path BloodMNIST/L16_WD01_asinh_R18_128/best_modules_boost_0/best_progs_code/1_13.jl \
  --model gemini-3.1-pro-preview
  --image-paths ../medmnist/2D/BloodMNIST/train/000000_label7.png ../medmnist/2D/BloodMNIST/train/000001_label3.png ../medmnist/2D/BloodMNIST/train/000002_label6.png
```

Optional reference images can also be provided with `--image-paths`. We just picked the first 3. 

## Plot generation

The Pareto plotting scripts are documented above. The remaining plots used in the paper are generated with the scripts below.

### Aggregating LLM scores

Before generating the LLM summary plots, aggregate the raw JSON files with:

```bash
python3 aggregate_llm_scores.py \
  --input-root . \
  --llms gemini openai anthropic
```

### LLM score distributions

The unigram and bigram LLM score distribution plots are generated with:

```bash
julia --project plot_llm_stats.jl \
  --aggregated_root=aggregated \
  --output_dir=plots
```

### Program-score distributions

The program-score distributions and the program-length histogram are generated from the reader CSVs with:

```bash
julia --project plot_program_score_stats.jl \
  --stats_file=BloodMNIST/L16_WD01_asinh_R18_128/best_modules_boost_0/asinh/best_modules_program_stats_beststrue_false.csv \
  --output_dir=plots \
  --score_plot=density \
  --program_length_plot=hist
```

### Sankey plots

The audience-to-audience Sankey plots are generated from the aggregated global LLM scores with:

```bash
python3 plot_llm_sankey.py \
  --aggregated_root aggregated/global \
  --output_dir plots
```

### Lollipop plots

The unigram lollipop plots are generated with:

```bash
python3 plot_llm_lollipop.py \
  --mode=llm_compare \
  --llm=gemini \
  --input_root=. \
  --output_dir=plots
```

## Per-class sensitivity and specificity

`plots/per_class_metrics.jl` builds the per-class table used in the appendix. Rates are
computed with `StatisticalMeasures.MulticlassTruePositiveRate` and
`MulticlassTrueNegativeRate` under `average = NoAvg()`, so the cells are sensitivity
(recall) and specificity, one versus rest, on the test split.

```bash
julia --project plots/per_class_metrics.jl \
  --output_dir BloodMNIST \
  --trial_id L16_WD01_asinh_R18_128 \
  --dataset BloodMNIST \
  --num_classes 8 \
  --nd 3 \
  --nn true \
  --metric macro_f1 \
  --class_names "basophil,eosinophil,erythroblast,immature gran.,lymphocyte,monocyte,neutrophil,platelet"
```

It writes `per_class_sens_spec_<metric>.csv` and a matching `.tex` next to the other ML
outputs. The columns are assembled from three sources.

- The nine distilled columns come from the `sensitivity` and `specificity` fields of
  `extended_metrics_<metric>_<suffix>.csv`, which `src/train_ml.jl` already writes.
- `--nn true` scores the three neural baselines. It loads the checkpoint with the best
  validation balanced accuracy through `python_utils/medmnist_resnet_baseline.py` and
  writes `per_class_test.csv` in each baseline folder. Use `--nd 1` for the grayscale
  datasets.
- The end-to-end MAGE column is read from `per_class_test_allfalse.csv`, or
  `per_class_test_alltrue.csv` if that is the one present, under
  `<output_dir>/MAGE_ALONE_20H/mage_imgcls/`, which `src/read_best_magecls_per_trial.jl`
  writes for the validation-selected individual. Rerun that reader once to produce it.
  `--all=false` is enough here, since only the validation-selected program is reported.

Any source that is missing is rendered as `--`, so the table can be produced
incrementally.

## Reproducibility of the archived runs

MAGE commit `dbe5e36` can be used to run all experiments

## Licenses and external assets

This repository uses the following external assets:

- MedMNIST datasets: used under the MedMNIST dataset licenses. The datasets used in this paper are BloodMNIST, PathMNIST, OCTMNIST, OrganAMNIST, and OrganCMNIST.
- torchvision pretrained models: used through torchvision under its open-source license.
- MAGE: used under its open-source license.
- LLM APIs: GPT-5, Claude Sonnet 4.6, and Gemini 3.1 Pro Preview were accessed through their providers' API terms.

The code introduced in this repository is released under the MIT License.
