# LLM Plots

This folder contains two plotting entry points:

- `plot_llm_stats.jl`
  It makes the Makie PDF plots for score distributions and program lengths.
- `plot_llm_sankey.py`
  It makes the Sankey diagram for the global score flow `cs -> doctor -> patient`.
- `plot_llm_lollipop.py`
  It makes unigram lollipop plots comparing the three audiences, split by library.
- `analyze_missing_llm_units.py`
  It compares raw LLM JSON files against the canonical unigram and bigram unit lists and reports how many units were not scored.

## 1. Makie score plots

This script reads:

- aggregated LLM score JSON files under `aggregated`
- optionally one per-program stats CSV produced by `read_best_models_per_trial.jl`

It writes PDF plots to `plots` by default.

### Default call

```bash
julia --project plot_llm_stats.jl \
  --aggregated_root=aggregated \
  --stats_file=BloodMNIST/L16_WD01_asinh_R18_128/best_modules_boost_0/asinh/best_modules_program_stats_bestsfalse_false.csv \
  --output_dir=plots
```

### Plot options

For the score distributions:

- `--score_plot=density`
- `--score_plot=hist`

For the program-length plot:

- `--program_length_plot=hist`
- `--program_length_plot=density`

Example:

```bash
julia --project plot_llm_stats.jl \
  --aggregated_root=aggregated \
  --stats_file=BloodMNIST/L16_WD01_asinh_R18_128/best_modules_boost_0/asinh/best_modules_program_stats_bestsfalse_false.csv \
  --output_dir=plots \
  --score_plot=density \
  --program_length_plot=hist
```

### Files produced

The Julia script writes:

- `global_unigram_audiences_density.pdf`
- `global_bigram_audiences_density.pdf`
- `cs_unigram_llms_density.pdf`
- `cs_bigram_llms_density.pdf`
- `doctor_unigram_llms_density.pdf`
- `doctor_bigram_llms_density.pdf`
- `patient_unigram_llms_density.pdf`
- `patient_bigram_llms_density.pdf`
- `program_length_hist.pdf`

The exact suffix changes if you use `--score_plot=hist` or `--program_length_plot=density`.

## 2. Sankey plot

This script reads the global aggregated JSON files and writes one interactive HTML Sankey diagram.

### Unigram

```bash
python3 plot_llm_sankey.py \
  --aggregated_root=aggregated/global \
  --gram=unigram \
  --output_dir=plots
```

### Bigram

```bash
python3 plot_llm_sankey.py \
  --aggregated_root=aggregated/global \
  --gram=bigram \
  --output_dir=plots
```

### Files produced

- `sankey_unigram.html`
- `sankey_bigram.html`

## Notes

- The Julia script expects the aggregated JSON schema already produced by `aggregate_llm_scores.py`.
- The Sankey script rounds the averaged scores to integers in `[1, 10]` before building the flow categories.
- If you do not want the program-length plot, omit `--stats_file`.

## 4. Missing-unit analysis

This script reads the raw LLM JSON files and compares them against:

- `symbol_descriptions.md` for unigrams
- `symbol_compositions.md` for bigrams

Example:

```bash
python3 analyze_missing_llm_units.py \
  --input-root=. \
  --llms gemini openai anthropic \
  --output-dir=analysis
```

Files produced:

- `llm_missing_units_per_group.csv`
- `llm_missing_units_per_file.csv`
- `llm_missing_units_detail.csv`

## 3. Unigram lollipop plots

For one LLM, comparing audiences:

```bash
python3 plot_llm_lollipop.py \
  --mode=llm_compare \
  --llm=chatgpt \
  --input_root=. \
  --output_dir=plots
```

For global averages, comparing audiences:

```bash
python3 plot_llm_lollipop.py \
  --mode=global_compare \
  --aggregated_root=aggregated \
  --output_dir=plots
```

Optional:

- `--top_n=80`
- `--per_page=40`
- `--fig_width=12`
- `--row_height=0.2`
