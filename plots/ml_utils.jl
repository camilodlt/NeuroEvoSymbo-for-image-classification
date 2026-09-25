include(joinpath(dirname(dirname(@__FILE__)), "utils", "ml_preprocessing.jl"))
using CSV

## ARGUMENTS ##

"""Add the common pareto-script CLI arguments shared by the model-specific plot drivers."""
function add_common_pareto_args!(settings::ArgParseSettings)
    @add_arg_table settings begin
        "--val"
        arg_type = Bool
        default = true
        "--test"
        arg_type = Bool
        default = true
        "--trial_id"
        arg_type = String
        "--output_dir"
        arg_type = String
        "--boost_round"
        arg_type = Int
        "--every_early"
        arg_type = Int
        default = 1
        "--every_late"
        arg_type = Int
        default = 1
        "--sweep_each_experiment"
        arg_type = Bool
        default = false
        "--quiet_column_corrections"
        arg_type = Bool
        default = true
        "--use_only_bests"
        arg_type = Bool
        "--use_really_all"
        arg_type = Bool
        default = false
        "--normalize"
        arg_type = Bool
        default = false
        "--act"
        arg_type = String
        default = "identity"
        "--pool_fn"
        arg_type = String
        default = ""
        "--second_trial"
        arg_type = String
        default = ""
        "--save_name"
        arg_type = String
        default = ""
        "--bar"
        arg_type = Int
        action = :append_arg
        default = Int[]
        "--resnet_val_bacc"
        arg_type = Float64
        default = NaN
        "--resnet_test_bacc"
        arg_type = Float64
        default = NaN
        "--mage_val_bacc"
        arg_type = Float64
        default = NaN
        "--mage_test_bacc"
        arg_type = Float64
        default = NaN
        "--stats_file"
        arg_type = String
        default = ""
        "--audience"
        arg_type = String
        default = ""
        "--gram"
        arg_type = String
        default = ""
        "--reducer"
        arg_type = String
        default = "mean"
        "--interpretability_bars"
        arg_type = Bool
        default = true
        "--show_bar_labels"
        arg_type = Bool
        default = false
        "--color_bars"
        arg_type = Bool
        default = true
    end
    return settings
end

## PATHS ##

const VAL_COLOR = "#E69F00"
const TEST_COLOR = "#0072B2"
const INTERP_COLORMAP = CairoMakie.Reverse(:cividis)

"""Resolve one dataset file path with support for legacy flat files and newer act/pool subfolders."""
function dataset_file_path(base_dir::String, split::String, suffix::String; act::String, pool_fn::String)
    filename = "best_modules_dataset_$(split)_bests$(suffix).jld2"
    if !isempty(pool_fn)
        return joinpath(base_dir, act, pool_fn, filename)
    end
    act_path = joinpath(base_dir, act, filename)
    legacy_path = joinpath(base_dir, filename)
    return isfile(act_path) ? act_path : legacy_path
end

"""Build the train, validation, and test dataset paths for all selected trials and boost rounds."""
function build_paths(parsed_args, suffix, has_val, has_test)
    @info "Building dataset paths" suffix has_val has_test act = parsed_args["act"] pool_fn = parsed_args["pool_fn"]
    train_paths = String[]
    val_paths = String[]
    test_paths = String[]

    function append_trial_paths!(trial_id::String)
        @info "Appending paths for trial" trial_id rounds = (0:(parsed_args["boost_round"]))
        for round in 0:(parsed_args["boost_round"])
            p = joinpath(parsed_args["output_dir"], trial_id, "best_modules_boost_$round")
            @info "Resolved boost directory" trial_id round path = p
            push!(train_paths, dataset_file_path(p, "train", suffix; act = parsed_args["act"], pool_fn = parsed_args["pool_fn"]))
            has_val && push!(val_paths, dataset_file_path(p, "val", suffix; act = parsed_args["act"], pool_fn = parsed_args["pool_fn"]))
            has_test && push!(test_paths, dataset_file_path(p, "test", suffix; act = parsed_args["act"], pool_fn = parsed_args["pool_fn"]))
        end
    end

    append_trial_paths!(parsed_args["trial_id"])
    parsed_args["second_trial"] != "" && append_trial_paths!(parsed_args["second_trial"])

    @info "Train paths: $(train_paths)"
    @info "Validation paths: $(val_paths)"
    @info "Test paths: $(test_paths)"
    @info "Path counts" train = length(train_paths) val = length(val_paths) test = length(test_paths)
    return train_paths, val_paths, test_paths
end

"""Normalize the requested save name into the final output path under output_dir/trial_id."""
function normalize_save_path(name::String, output_dir::String, trial_id::String)
    filename = isempty(strip(name)) ? "$(trial_id)_pareto_plot.png" : name
    base_dir = joinpath(output_dir, trial_id)
    out = isabspath(filename) ? filename : joinpath(base_dir, filename)
    normalized = splitext(out)[2] == "" ? out * ".png" : out
    @info "Normalized save path" requested = name base_dir resolved = normalized
    return normalized
end

"""Return the output path used for the companion plot with a log-scaled x axis."""
function log_variant_path(path::String)
    root, ext = splitext(path)
    return root * "_log" * ext
end

"""Return the CSV path used to save sweep-curve data next to a figure output."""
function curve_data_path(path::String)
    root, _ext = splitext(path)
    return root * "_curve.csv"
end

## LOAD DATA ##

"""Load several saved prediction datasets and merge them into one matrix with a shared target vector."""
function load_xy(paths::Vector{String})
    @info "Loading prediction matrices and labels" n_paths = length(paths)
    xs = []
    ys = []
    for (idx, path) in enumerate(paths)
        @info "Reading Data at $path"
        data = JLD2.load(path)["single_stored_object"]
        @info "Loaded dataset object" path idx ys_length = length(data.ys) gt_length = length(data.gt)
        push!(xs, data.ys)
        push!(ys, data.gt)
    end
    x = reduce(hcat, map(v -> reduce(vcat, v'), xs))
    @assert all(all(ys[1] .== ys[i]) for i in eachindex(ys)) "Loaded datasets did not share the same labels"
    y = ys[1]
    @info "Merged arrays" x_size = size(x) y_size = size(y) n_sources = length(ys)
    return x, y
end

"""Print one UnicodePlots histogram to summarize the label distribution of one split."""
function print_label_histogram(name::String, y)
    @info "Printing label histogram" dataset = name n = length(y) unique_labels = unique(y)
    println("\n$name label histogram")
    println(histogram(y; title = name, xlabel = "Label", ylabel = "Count"))
    return nothing
end

## PREPROCESS ##

"""Correct one prediction column from encoded labels into the numeric convention used downstream."""
function correct_ys!(ys::AbstractVector; quiet::Bool = true)
    ys_unique = unique(ys)
    n_unique = length(ys_unique)
    has_one = 1.0 in ys_unique
    has_two = 2.0 in ys_unique

    if n_unique == 2 && has_one && has_two
        !quiet && @info "Correcting binary prediction column from {1,2} to {-1,1}"
        ys[ys .== 1.0] .= -1.0
        ys[ys .== 2.0] .= 1.0
    elseif n_unique == 3
        !quiet && @info "Correcting ternary prediction column from {1,2,3} to {-1,0,1}"
        ys .-= 2
    else
        !quiet && @warn "Problem with column that had unique $ys_unique"
    end
    return ys
end

"""Convert one prediction matrix into a DataFrame and coerce discrete count columns to continuous."""
function coerce_prediction_matrix(x)
    @info "Converting matrix to DataFrame and coercing numeric columns" input_size = size(x)
    mat = DataFrame(x, :auto)
    coerce!(mat, Count => Continuous)
    @info "Coercion done" output_size = size(mat)
    return mat
end

"""Create an internal validation split only when the user did not provide one on disk."""
function maybe_make_val_split(trainx, trainy, has_val)
    if has_val
        @info "Validation set provided by input files"
        return nothing
    end
    @info "Since no val data, making own val split at 0.8"
    return MLUtils.splitobs((trainx, trainy); at = 0.8, shuffle = true, stratified = trainy)
end

"""Drop duplicate predictor columns while preserving alignment across train, validation, and test matrices."""
function drop_duplicate_columns!(train_mat, val_mat, test_mat)
    @info "Checking duplicate predictor columns" train_cols = ncol(train_mat)
    unique_train_data = unique(last, pairs(eachcol(train_mat)))
    unique_train_cols = string.(first.(unique_train_data))
    dropped_cols = setdiff(names(train_mat), unique_train_cols)

    if isempty(dropped_cols)
        @info "No duplicate columns detected"
        return train_mat, val_mat, test_mat
    end

    @info "Number of dropped cols $(length(dropped_cols))"
    @info "Dropped column names" dropped_cols
    select!(train_mat, Not(dropped_cols))
    select!(val_mat, Not(dropped_cols))
    !isnothing(test_mat) && select!(test_mat, Not(dropped_cols))
    @info "Shapes after dropping duplicates" train_size = size(train_mat) val_size = size(val_mat) test_size = isnothing(test_mat) ? nothing : size(test_mat)
    return train_mat, val_mat, test_mat
end

"""Return 1-based indices of the first occurrence of each unique prediction column."""
function unique_column_indices(mat::DataFrame)
    unique_pairs = unique(last, pairs(eachcol(mat)))
    kept_names = string.(first.(unique_pairs))
    kept_idx = indexin(kept_names, names(mat))
    @assert all(.!isnothing.(kept_idx)) "Could not recover kept indices from unique column names"
    return Int[i for i in kept_idx]
end

"""Infer the header CSV path that matches one exported prediction dataset."""
function dataset_header_path(dataset_path::String)
    base = basename(dataset_path)
    m = match(r"^best_modules_dataset_(?:train|val|test)_bests(.+)\.jld2$", base)
    isnothing(m) && error("Could not infer header filename from dataset path: $dataset_path")
    suffix = m.captures[1]
    return joinpath(dirname(dataset_path), "best_modules_header_bests$(suffix).csv")
end

"""Resolve the stats CSV path for one dataset directory from the user-provided template path."""
function dataset_stats_path(dataset_path::String, stats_file::String)
    stats_file == "" && return nothing
    if isfile(stats_file)
        candidate = joinpath(dirname(dataset_path), basename(stats_file))
        return isfile(candidate) ? candidate : stats_file
    end
    return stats_file
end

"""Build the interpretability/stat-column name requested by audience, gram, and reducer."""
function stats_column_name(audience::String, gram::String, reducer::String)
    @assert reducer in ("mean", "min") "Argument --reducer must be mean or min"
    @assert gram in ("unigram", "bigram") "Argument --gram must be unigram or bigram"
    @assert audience != "" "Argument --audience is required when --stats_file is provided"
    suffix = gram == "unigram" ? "_unigram_avg_single" : "_bigram_avg_composition1"
    return Symbol("$(reducer)_$(audience)$(suffix)")
end

"""Load one stats column per exported prediction column by aligning stats rows to header rows."""
function load_stats_scores_per_column(train_paths::Vector{String}, stats_file::String, score_col::Symbol)
    stats_values = Float64[]
    for dataset_path in train_paths
        header_path = dataset_header_path(dataset_path)
        @assert isfile(header_path) "Missing header CSV for dataset path $dataset_path at $header_path"
        header_df = CSV.read(header_path, DataFrame)
        sort!(header_df, :column_idx)
        @assert header_df.column_idx == collect(1:nrow(header_df)) "Header column_idx in $header_path must be contiguous from 1 to nrow"

        stats_path = dataset_stats_path(dataset_path, stats_file)
        @assert !isnothing(stats_path) && isfile(stats_path) "Missing stats CSV for dataset path $dataset_path"
        stats_df = CSV.read(stats_path, DataFrame)
        score_col_name = String(score_col)
        @assert score_col_name in names(stats_df) "Requested stats column $score_col not found in $stats_path"

        left_cols = [:checkpoint_path, :checkpoint_name, :module_key, :output_idx, :activation, :seed_root]
        joined = leftjoin(header_df, stats_df[:, vcat(left_cols, [score_col])], on = left_cols)
        @assert nrow(joined) == nrow(header_df) "Header/stats join changed row count for $dataset_path"
        @assert all(.!ismissing.(joined[!, score_col])) "Missing joined stats values in $stats_path for column $score_col"
        append!(stats_values, Float64.(joined[!, score_col]))
    end
    return stats_values
end

"""Compute vertical-bar locations when the sorted interpretability score drops to a new value."""
function interpretability_transition_bars(sorted_scores::Vector{Float64})
    bars = Int[]
    labels = Dict{Int, String}()
    bucket_scores = Dict{Int, Float64}()
    isempty(sorted_scores) && return bars, labels, bucket_scores

    current_bucket = floor(sorted_scores[1] * 2) / 2
    for i in 2:length(sorted_scores)
        next_bucket = floor(sorted_scores[i] * 2) / 2
        if next_bucket < current_bucket
            push!(bars, i - 1)
            labels[i - 1] = string(current_bucket)
            bucket_scores[i - 1] = current_bucket
            current_bucket = next_bucket
        end
    end
    return bars, labels, bucket_scores
end

"""Return y-axis tick values from the floored minimum up to 1.0 using a fixed step."""
function ytick_values_for_bacc(vals::Vector{Float64}; step::Float64 = 0.05, upper::Float64 = 1.0)
    finite_vals = filter(isfinite, vals)
    isempty(finite_vals) && return 0.0:step:upper
    lower = floor(minimum(finite_vals) * 10) / 10
    return lower:step:upper
end

"""Return the program ordering induced by interpretability first and importance second, or by importance only."""
function rank_programs(importances::Vector{Float64}, program_names::Vector{String}; interpret_scores::Union{Nothing, Vector{Float64}} = nothing)
    if isnothing(interpret_scores)
        order = sortperm(importances, rev = true)
        return order, Dict{Int, String}(), Dict{Int, Float64}()
    end
    @assert length(interpret_scores) == length(importances) == length(program_names) "Ranking vectors must align"
    order = sortperm(1:length(importances); by = i -> (interpret_scores[i], importances[i]), rev = true)
    sorted_scores = interpret_scores[order]
    bars, labels, bucket_scores = interpretability_transition_bars(sorted_scores)
    return order, labels, bucket_scores
end

"""Append ranking metadata to the export path so saved figures and CSVs are audience/gram/reducer/suffix aware."""
function decorate_export_path(path::String; algorithm::String, suffix::String, stats_file::String, audience::String, gram::String, reducer::String)
    _root, ext = splitext(path)
    base_name = basename(path)
    trial_root = dirname(path)
    target_dir = if isempty(stats_file)
        joinpath(trial_root, "plots", suffix, "fi_only", "none")
    else
        joinpath(trial_root, "plots", suffix, audience, reducer)
    end
    mkpath(target_dir)
    tag = if isempty(stats_file)
        "$(algorithm)_fi_only"
    else
        "$(algorithm)_$(audience)_$(reducer)_$(gram)"
    end
    return joinpath(target_dir, splitext(base_name)[1] * "_" * tag * ext)
end

"""Write the top-k sweep data to CSV so the global overlay plot can reuse it without recomputing."""
function save_curve_data(
        outpath::String,
        xs,
        val_front,
        val_raw,
        test_selected,
        test_raw;
        algorithm::String,
        trial_id::String,
        suffix::String,
        act::String,
        audience::String,
        gram::String,
        reducer::String,
        stats_file::String,
        rank_mode::String,
    )
    df = DataFrame(
        k = xs,
        val_front = val_front,
        val_raw = val_raw,
        test_selected = test_selected,
        test_raw = test_raw,
    )
    df[!, :algorithm] .= algorithm
    df[!, :trial_id] .= trial_id
    df[!, :suffix] .= suffix
    df[!, :act] .= act
    df[!, :audience] .= audience
    df[!, :gram] .= gram
    df[!, :reducer] .= reducer
    df[!, :stats_file] .= stats_file
    df[!, :rank_mode] .= rank_mode
    csv_path = curve_data_path(outpath)
    CSV.write(csv_path, df)
    @info "Saved curve data" csv_path n_rows = nrow(df)
    return csv_path
end

"""Apply prediction-column corrections to every non-nothing DataFrame that is passed in."""
function correct_prediction_columns!(dfs...; quiet::Bool = true)
    for (df_idx, df) in enumerate(dfs)
        isnothing(df) && continue
        @info "Correcting prediction columns in dataset" dataset_index = df_idx ncols = ncol(df)
        for c in 1:ncol(df)
            !quiet && @info "Correcting column" dataset_index = df_idx column = c unique_values = unique(df[:, c])
            correct_ys!(@view df[:, c]; quiet = quiet)
        end
    end
    return nothing
end

## METRICS ##

"""Compute balanced accuracy for one fitted MLJ machine on one feature matrix and target vector."""
function calculate_bacc(mach, x, y; quiet::Bool = false)
    !quiet && @info "Calculating balanced accuracy" x_size = size(x) y_size = size(y)
    y_hat = predict_mode(mach, x)
    bacc = StatisticalMeasures.balanced_accuracy(y_hat, y)
    !quiet && @info "Balanced accuracy computed" bacc
    return bacc
end

## SWEEP ##

"""Build the top-k evaluation grid using one step before 100 programs and another after 100."""
function build_step_grid(n_programs::Int, every_early::Int, every_late::Int; split_at::Int = 100)
    early_end = min(split_at, n_programs)
    early = collect(1:every_early:early_end)
    late_start = split_at + every_late
    late = late_start <= n_programs ? collect(late_start:every_late:n_programs) : Int[]
    xs = unique(vcat(early, late))
    @info "Built top-k step grid" n_programs every_early every_late split_at n_points = length(xs)
    return xs
end

## PLOT ##

"""Build the CairoMakie font theme used for LaTeX-styled figures with a CMU Serif fallback."""
function make_font_theme()
    if isdefined(CairoMakie.Makie, :theme_latexfonts)
        @info "Using Makie LaTeX font theme"
        return CairoMakie.Makie.theme_latexfonts()
    end

    @info "Makie LaTeX theme helper unavailable, falling back to CMU Serif fonts"
    return CairoMakie.Theme(
        fonts = (;
            regular = "CMU Serif",
            bold = "CMU Serif Bold",
            italic = "CMU Serif Italic",
            bold_italic = "CMU Serif Bold Italic",
        ),
    )
end

"""Return whether one optional baseline metric was provided by the caller."""
has_metric(x) = !isnan(x)

"""Draw one horizontal baseline line only when the corresponding metric value is available."""
function plot_optional_baseline!(ax, value::Float64; color, linestyle)
    if has_metric(value)
        @info "Adding baseline line" value color linestyle
        CairoMakie.hlines!(ax, [value], color = color, linewidth = 3, linestyle = linestyle)
    else
        @info "Skipping baseline line because value was not provided" value color linestyle
    end
    return nothing
end

"""Draw the interpretability-score label for each bar, stacking labels onto higher rows (fixed pixel offsets) whenever neighboring bars are too close together in x to avoid overlapping text."""
function place_bar_labels!(ax, bars::Vector{Int}, bar_labels::Dict{Int, String}; xscale, y::Float64, row_height_px::Float64 = 20.0, min_gap_frac::Float64 = 0.07, axis_span::Float64 = NaN)
    labeled_bars = filter(b -> haskey(bar_labels, b), bars)
    isempty(labeled_bars) && return nothing

    txs = xscale.(labeled_bars)
    order = sortperm(txs)
    # The gap below which two labels would collide is a property of the axis, not of the bars.
    # Measuring it against the bars' own span lets a tightly clustered set shrink the threshold
    # with itself, so every label is judged far enough apart and they all land on one row.
    span = isnan(axis_span) ? maximum(txs) - minimum(txs) : axis_span
    min_gap = span > 0 ? min_gap_frac * span : Inf

    row_last_x = Float64[]
    for i in order
        b, tx = labeled_bars[i], txs[i]
        row = findfirst(r -> (tx - row_last_x[r]) >= min_gap, eachindex(row_last_x))
        if isnothing(row)
            push!(row_last_x, tx)
            row = length(row_last_x)
        else
            row_last_x[row] = tx
        end
        CairoMakie.text!(
            ax, b, y;
            text = bar_labels[b],
            align = (:center, :bottom),
            offset = (0, 2 + (row - 1) * row_height_px),
            fontsize = 19,
            font = :bold,
        )
    end
    return nothing
end

"""Draw and save one pareto-style plot with grouped legends, baseline overlays, and optional vertical bars."""
function plot_front(
        xs,
        val_front,
        val_raw,
        test_selected,
        test_raw,
        outpath,
        title,
        baselines;
        has_test::Bool,
        model_label::String,
        bars::Vector{Int} = Int[],
        bar_labels::Dict{Int, String} = Dict{Int, String}(),
        bar_scores::Dict{Int, Float64} = Dict{Int, Float64}(),
        show_bar_labels::Bool = false,
        color_bars::Bool = true,
        xscale = identity,
    )
    @info "Preparing plot" outpath title n_points = length(xs) has_test baselines model_label bars show_bar_labels color_bars xscale
    CairoMakie.with_theme(make_font_theme()) do
        fig = CairoMakie.Figure(size = (1380, 700))
        ax = CairoMakie.Axis(
            fig[1, 1],
            xlabel = L"\mathrm{Programs\ used}",
            ylabel = L"\mathrm{Balanced\ accuracy}",
            title = title,
            xscale = xscale,
            xgridvisible = false,
            ygridvisible = true,
            xlabelsize = 28,
            ylabelsize = 28,
            xticklabelsize = 20,
            yticklabelsize = 20,
            titlealign = :left,
        )

        yticks = ytick_values_for_bacc(vcat(val_front, val_raw, test_selected, test_raw))
        ax.yticks = yticks

        if !isempty(bars)
            @info "Adding vertical bars" bars color_bars
            color_kwargs = if color_bars
                bar_values = [get(bar_scores, bar, 1.0) for bar in bars]
                (color = bar_values, colormap = INTERP_COLORMAP, colorrange = (1.0, 10.0))
            else
                (color = :gray40,)
            end
            CairoMakie.vlines!(
                ax,
                bars;
                color_kwargs...,
                linewidth = 3.2,
                linestyle = :dashdot,
                depth_shift = -1.0f0,
            )
            if show_bar_labels
                place_bar_labels!(ax, bars, bar_labels; xscale = xscale, y = first(yticks), axis_span = Float64(xscale(maximum(xs)) - xscale(minimum(xs))))
            end
        end

        val_elbows = [i == 1 || val_front[i] > val_front[i - 1] for i in eachindex(val_front)]
        @info "Detected validation elbow points" n_elbows = count(val_elbows)
        CairoMakie.lines!(ax, xs, val_front, color = VAL_COLOR, linewidth = 4)
        CairoMakie.scatter!(ax, xs[val_elbows], val_front[val_elbows], color = VAL_COLOR, marker = :xcross, markersize = 13)
        CairoMakie.scatter!(ax, xs, val_raw, color = VAL_COLOR, alpha = 0.25, markersize = 7)

        if has_test
            @info "Detected test elbow points from validation elbows" n_elbows = count(val_elbows)
            CairoMakie.lines!(ax, xs, test_selected, color = TEST_COLOR, linewidth = 4)
            CairoMakie.scatter!(ax, xs[val_elbows], test_selected[val_elbows], color = TEST_COLOR, marker = :xcross, markersize = 13)
            CairoMakie.scatter!(ax, xs, test_raw, color = TEST_COLOR, alpha = 0.25, markersize = 7)
        end

        plot_optional_baseline!(ax, baselines.resnet_val; color = VAL_COLOR, linestyle = :dot)
        has_test && plot_optional_baseline!(ax, baselines.resnet_test; color = TEST_COLOR, linestyle = :dot)
        plot_optional_baseline!(ax, baselines.mage_val; color = VAL_COLOR, linestyle = :dash)
        has_test && plot_optional_baseline!(ax, baselines.mage_test; color = TEST_COLOR, linestyle = :dash)

        split_elements = [
            CairoMakie.LineElement(color = VAL_COLOR, linewidth = 4),
            CairoMakie.LineElement(color = TEST_COLOR, linewidth = 4),
        ]
        split_labels = ["Validation", "Test"]
        model_elements = [
            CairoMakie.MarkerElement(color = :black, marker = :xcross, markersize = 16),
            CairoMakie.LineElement(color = :black, linewidth = 4, linestyle = :dot),
            CairoMakie.LineElement(color = :black, linewidth = 4, linestyle = :dash),
        ]
        model_labels = [model_label, "ResNet", "MAGE Alone"]
        @info "Building grouped legend" split_labels model_labels
        CairoMakie.Legend(
            fig[1, 2],
            [split_elements, model_elements],
            [split_labels, model_labels],
            ["Data Split", "Models"],
            tellheight = false,
            tellwidth = true,
            labelsize = 18,
            titlesize = 20,
            rowgap = 12,
            patchsize = (40, 20),
        )
        if !isempty(bars) && color_bars
            CairoMakie.Colorbar(
                fig[1, 3],
                limits = (1.0, 10.0),
                colormap = INTERP_COLORMAP,
                label = "Interpretability",
                vertical = true,
                flipaxis = false,
                ticklabelsize = 18,
                labelsize = 20,
                ticks = 1:10,
            )
        end
        @info "Saving plot figure" outpath
        CairoMakie.save(outpath, fig)
    end
    @info "Plot saved successfully" outpath
    return nothing
end
