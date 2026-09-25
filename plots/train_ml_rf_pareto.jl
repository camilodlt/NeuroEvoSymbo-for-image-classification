using ArgParse
using MLJ
using JLD2
using DataFrames
using Statistics
using MLUtils
using UnicodePlots
using Pkg
using Logging
using PythonCall

file = @__FILE__
home = dirname(dirname(file))
include(joinpath(home, "utils", "datasets.jl"))
@info "Loaded dataset utilities from $(joinpath(home, "utils", "datasets.jl"))"
RandomForestClassifierSKLEARN = MLJ.@load RandomForestClassifier pkg = MLJScikitLearnInterface verbosity = 0

Pkg.activate(@__DIR__)
using CairoMakie
using LaTeXStrings
include(joinpath(@__DIR__, "ml_utils.jl"))

function build_parser()
    @info "Building argument parser"
    s = ArgParseSettings()
    add_common_pareto_args!(s)
    @add_arg_table s begin
        "--importance_method"
        arg_type = String
        default = "feature_importance"
        "--permutation_repeats"
        arg_type = Int
        default = 10
    end
    @info "Argument parser ready"
    return s
end

function build_rf_model(;
        max_depth::Int,
        min_samples_split::Int,
        max_samples::Float64,
        min_samples_leaf::Int,
        max_features,
        criterion::String,
        class_weight::String,
        log_creation::Bool = true,
    )
    params = (
        n_estimators = 300,
        max_depth = max_depth,
        class_weight = class_weight,
        criterion = criterion,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        max_samples = max_samples,
        n_jobs = -1,
        random_state = 1,
    )
    log_creation && @info "Creating RF model" params
    return RandomForestClassifierSKLEARN(
        n_estimators = 300,
        max_depth = max_depth,
        class_weight = class_weight,
        criterion = criterion,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        max_samples = max_samples,
        n_jobs = -1,
        random_state = 1,
    )
end

function fit_rf_machine(
        train_mat,
        trainy_cat;
        max_depth::Int,
        min_samples_split::Int,
        max_samples::Float64,
        min_samples_leaf::Int,
        max_features,
        criterion::String,
        class_weight::String,
        log_low_level::Bool,
    )
    model = build_rf_model(;
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        max_samples = max_samples,
        min_samples_leaf = min_samples_leaf,
        max_features = max_features,
        criterion = criterion,
        class_weight = class_weight,
        log_creation = log_low_level,
    )
    mach = machine(model, train_mat, trainy_cat)
    return MLJ.fit!(mach; verbosity = log_low_level ? 1 : 0)
end

function select_best_rf_params(train_mat, trainy, val_mat, valy; test_mat = nothing, testy = nothing, log_candidates::Bool = false, label = "ranking model")
    sklearn = pyimport("sklearn")
    builtins = pyimport("builtins")
    np = pyimport("numpy")
    model_selection = sklearn.model_selection
    ensemble = sklearn.ensemble

    x_train = Matrix(train_mat)
    x_val = Matrix(val_mat)
    y_train = Int.(trainy)
    y_val = Int.(valy)

    x_all_np = np.asarray(vcat(x_train, x_val))
    y_all_np = np.asarray(vcat(y_train, y_val))
    test_fold_np = np.asarray(vcat(fill(-1, size(x_train, 1)), fill(0, size(x_val, 1))))
    ps = model_selection.PredefinedSplit(test_fold_np)

    param_grid = builtins.dict(
        max_depth = [10, 20, 30],
        min_samples_split = [5, 10, 20],
        max_samples = [0.6, 0.7, 0.8],
        min_samples_leaf = [2, 5],
        max_features = ["sqrt", 0.5],
        criterion = ["gini"],
    )

    base_estimator = ensemble.RandomForestClassifier(
        n_estimators = 100,
        class_weight = "balanced",
        random_state = 1,
        n_jobs = 14,
    )

    @info "Starting RF GridSearchCV" label n_train = size(x_train, 1) n_val = size(x_val, 1) n_features = size(x_train, 2) n_estimators = 100
    grid = model_selection.GridSearchCV(
        estimator = base_estimator,
        param_grid = param_grid,
        scoring = "balanced_accuracy",
        cv = ps,
        refit = true,
        n_jobs = 1,
        verbose = 0,
    )
    grid.fit(x_all_np, y_all_np)
    best_params = pyconvert(Dict{String, Any}, grid.best_params_)
    best_val_bacc = pyconvert(Float64, grid.best_score_)
    @info "Finished RF GridSearchCV" label best_params best_val_bacc
    return (params = best_params, val_bacc = best_val_bacc)
end

function extract_rf_feature_importances(mach, feature_names)
    importances = mach |> feature_importances .|> last
    @assert length(importances) == length(feature_names) "Feature-importance length did not match number of columns"
    @info "Computed RF impurity-based importances" n_importances = length(importances) max_importance = maximum(importances) min_importance = minimum(importances)
    return importances
end

function extract_permutation_importances(mach, x, y; n_repeats::Int)
    @info "Computing permutation importances with sklearn" x_size = size(x) y_size = size(y) n_repeats
    sklearn_inspection = pyimport("sklearn.inspection")
    fp = fitted_params(mach)
    fitresult = hasproperty(fp, :fitresult) ? getproperty(fp, :fitresult) : nothing
    isnothing(fitresult) && error("Could not extract sklearn fitresult from fitted_params(mach)")
    x_matrix = Matrix(x)
    result = sklearn_inspection.permutation_importance(
        fitresult,
        x_matrix,
        collect(y);
        n_repeats = n_repeats,
        random_state = 1,
        scoring = "balanced_accuracy",
        n_jobs = -1,
    )
    importances = pyconvert(Vector{Float64}, result.importances_mean)
    @assert length(importances) == ncol(x) "Permutation-importance length did not match number of columns"
    @info "Computed permutation importances" n_importances = length(importances) max_importance = maximum(importances) min_importance = minimum(importances)
    return importances
end

function extract_importances(mach, x, y; method::String, n_repeats::Int)
    if method == "feature_importance"
        return extract_rf_feature_importances(mach, names(x))
    elseif method == "permutation"
        return extract_permutation_importances(mach, x, y; n_repeats = n_repeats)
    else
        error("Unknown importance method: $method. Expected one of: feature_importance, permutation")
    end
end

parsed_args = parse_args(build_parser())
@show parsed_args
@info "Parsed arguments" parsed_args

boost_round = parsed_args["boost_round"]
every_early = parsed_args["every_early"]
every_late = parsed_args["every_late"]
sweep_each_experiment = parsed_args["sweep_each_experiment"]
quiet_column_corrections = parsed_args["quiet_column_corrections"]
bars = sort(unique(parsed_args["bar"]))
importance_method = parsed_args["importance_method"]
permutation_repeats = parsed_args["permutation_repeats"]
stats_file = parsed_args["stats_file"]
audience = parsed_args["audience"]
gram = parsed_args["gram"]
reducer = parsed_args["reducer"]
interpretability_bars = parsed_args["interpretability_bars"]
show_bar_labels = parsed_args["show_bar_labels"]
color_bars = parsed_args["color_bars"]
use_bests = parsed_args["use_only_bests"]
use_really_all = parsed_args["use_really_all"]
has_val = parsed_args["val"]
has_test = parsed_args["test"]
suffix = "$(use_bests)_$(use_really_all)"
@assert every_early >= 1 "Argument --every_early must be at least 1"
@assert every_late >= 1 "Argument --every_late must be at least 1"
@assert importance_method in ("feature_importance", "permutation") "Argument --importance_method must be feature_importance or permutation"
@assert permutation_repeats >= 1 "Argument --permutation_repeats must be at least 1"
isempty(stats_file) || @assert !isempty(audience) && !isempty(gram) "When --stats_file is provided, --audience and --gram are required"
@info "Resolved run configuration" boost_round every_early every_late sweep_each_experiment quiet_column_corrections bars importance_method permutation_repeats stats_file audience gram reducer interpretability_bars show_bar_labels use_bests use_really_all has_val has_test suffix

baselines = (
    resnet_val = parsed_args["resnet_val_bacc"],
    resnet_test = parsed_args["resnet_test_bacc"],
    mage_val = parsed_args["mage_val_bacc"],
    mage_test = parsed_args["mage_test_bacc"],
)
@info "Resolved optional baselines" baselines
@info "ResNet baseline provided" val = has_metric(baselines.resnet_val) test = has_metric(baselines.resnet_test)
@info "MAGE Alone baseline provided" val = has_metric(baselines.mage_val) test = has_metric(baselines.mage_test)

train_paths, val_paths, test_paths = build_paths(parsed_args, suffix, has_val, has_test)

trainx, trainy = load_xy(train_paths)

if has_val
    valx, valy = load_xy(val_paths)
else
    (trainx, trainy), (valx, valy) = maybe_make_val_split(trainx, trainy, has_val)
end

testx, testy = nothing, nothing
if has_test
    testx, testy = load_xy(test_paths)
end

print_label_histogram("Train", trainy)
print_label_histogram("Validation", valy)
has_test && print_label_histogram("Test", testy)
@info "Finished loading raw arrays"

train_mat = coerce_prediction_matrix(trainx)
val_mat = coerce_prediction_matrix(valx)
test_mat = has_test ? coerce_prediction_matrix(testx) : nothing
@info "Constructed prediction DataFrames" train_size = size(train_mat) val_size = size(val_mat) test_size = isnothing(test_mat) ? nothing : size(test_mat)

interpret_scores_raw = nothing
if !isempty(stats_file)
    score_col = stats_column_name(audience, gram, reducer)
    interpret_scores_raw = load_stats_scores_per_column(train_paths, stats_file, score_col)
    @assert length(interpret_scores_raw) == ncol(train_mat) "Interpretability vector must align with original train columns"
    @info "Loaded interpretability stats for RF ranking" score_col n_scores = length(interpret_scores_raw) min_score = minimum(interpret_scores_raw) max_score = maximum(interpret_scores_raw)
end

kept_indices = unique_column_indices(train_mat)
train_mat, val_mat, test_mat = drop_duplicate_columns!(train_mat, val_mat, test_mat)
interpret_scores = isnothing(interpret_scores_raw) ? nothing : interpret_scores_raw[kept_indices]
correct_prediction_columns!(train_mat, val_mat, test_mat; quiet = quiet_column_corrections)
train_mat, val_mat, test_mat = maybe_standard_scale_prediction_matrices(
    train_mat,
    val_mat,
    test_mat;
    normalize = parsed_args["normalize"],
)
@info "Finished prediction post-processing" train_cols = ncol(train_mat) val_cols = ncol(val_mat) test_cols = isnothing(test_mat) ? nothing : ncol(test_mat)

trainy_cat = categorical(trainy)
@info "Converted training labels to categorical" levels = levels(trainy_cat)

@info "Selecting ranking RF on all $(ncol(train_mat)) programs"
best_ranking_rf = select_best_rf_params(
    train_mat,
    trainy,
    val_mat,
    valy;
    test_mat = test_mat,
    testy = testy,
    log_candidates = false,
    label = "ranking model",
)
best_rf_params = (
    max_depth = Int(best_ranking_rf.params["max_depth"]),
    min_samples_split = Int(best_ranking_rf.params["min_samples_split"]),
    max_samples = Float64(best_ranking_rf.params["max_samples"]),
    min_samples_leaf = Int(best_ranking_rf.params["min_samples_leaf"]),
    max_features = best_ranking_rf.params["max_features"],
    criterion = String(best_ranking_rf.params["criterion"]),
    class_weight = "balanced",
)
ranking_mach = fit_rf_machine(
    train_mat,
    trainy_cat;
    max_depth = best_rf_params.max_depth,
    min_samples_split = best_rf_params.min_samples_split,
    max_samples = best_rf_params.max_samples,
    min_samples_leaf = best_rf_params.min_samples_leaf,
    max_features = best_rf_params.max_features,
    criterion = best_rf_params.criterion,
    class_weight = best_rf_params.class_weight,
    log_low_level = false,
)
importance_x = importance_method == "permutation" ? val_mat : train_mat
importance_y = importance_method == "permutation" ? valy : trainy
importances = extract_importances(ranking_mach, importance_x, importance_y; method = importance_method, n_repeats = permutation_repeats)
program_order, _interp_bar_labels, _interp_bar_scores = rank_programs(importances, names(train_mat); interpret_scores = interpret_scores)
program_names = names(train_mat)[program_order]
@info "Using best RF parameters for top-k sweep" best_rf_params best_ranking_val_bacc = best_ranking_rf.val_bacc
top_idx = program_order[1:min(10, end)]
@info "Top ranked programs" top_programs = program_names[1:min(10, end)] top_importances = importances[top_idx] top_interpretability = isnothing(interpret_scores) ? nothing : interpret_scores[top_idx]

n_programs = length(program_order)
xs = build_step_grid(n_programs, every_early, every_late)
val_raw = Vector{Float64}(undef, length(xs))
val_front = Vector{Float64}(undef, length(xs))
test_raw = fill(NaN, length(xs))
test_selected = fill(NaN, length(xs))

best_val_so_far = -Inf
@info "Starting top-k RF sweep" n_programs every_early every_late n_points = length(xs)

for (i, k) in enumerate(xs)
    selected_programs = program_names[1:k]
    @info "Training RF for selected top-k programs" k first_program = selected_programs[1] last_program = selected_programs[end]

    params_for_k = best_rf_params
    if sweep_each_experiment
        best_rf_for_k = select_best_rf_params(
            train_mat[:, selected_programs],
            trainy,
            val_mat[:, selected_programs],
            valy;
            test_mat = has_test ? test_mat[:, selected_programs] : nothing,
            testy = testy,
            log_candidates = false,
            label = "top-k=$k",
        )
        params_for_k = (
            max_depth = Int(best_rf_for_k.params["max_depth"]),
            min_samples_split = Int(best_rf_for_k.params["min_samples_split"]),
            max_samples = Float64(best_rf_for_k.params["max_samples"]),
            min_samples_leaf = Int(best_rf_for_k.params["min_samples_leaf"]),
            max_features = best_rf_for_k.params["max_features"],
            criterion = String(best_rf_for_k.params["criterion"]),
            class_weight = "balanced",
        )
        mach = fit_rf_machine(
            train_mat[:, selected_programs],
            trainy_cat;
            max_depth = params_for_k.max_depth,
            min_samples_split = params_for_k.min_samples_split,
            max_samples = params_for_k.max_samples,
            min_samples_leaf = params_for_k.min_samples_leaf,
            max_features = params_for_k.max_features,
            criterion = params_for_k.criterion,
            class_weight = params_for_k.class_weight,
            log_low_level = false,
        )
    else
        mach = fit_rf_machine(
            train_mat[:, selected_programs],
            trainy_cat;
            max_depth = params_for_k.max_depth,
            min_samples_split = params_for_k.min_samples_split,
            max_samples = params_for_k.max_samples,
            min_samples_leaf = params_for_k.min_samples_leaf,
            max_features = params_for_k.max_features,
            criterion = params_for_k.criterion,
            class_weight = params_for_k.class_weight,
            log_low_level = false,
        )
    end

    val_raw[i] = calculate_bacc(mach, val_mat[:, selected_programs], valy)
    is_new_best_val = val_raw[i] > best_val_so_far
    global best_val_so_far = max(best_val_so_far, val_raw[i])
    val_front[i] = best_val_so_far

    if has_test
        test_raw[i] = calculate_bacc(mach, test_mat[:, selected_programs], testy)
        test_selected[i] = is_new_best_val ? test_raw[i] : (i == 1 ? test_raw[i] : test_selected[i - 1])
    end

    @info "Completed top-k RF fit" k step_index = i total_steps = length(xs) params_for_k val_bacc = val_raw[i] val_pareto = val_front[i] new_best_val = is_new_best_val test_bacc = test_raw[i] test_selected = test_selected[i]
end
@info "Finished top-k RF sweep"

save_path = normalize_save_path(parsed_args["save_name"], parsed_args["output_dir"], parsed_args["trial_id"])
save_path = decorate_export_path(save_path; algorithm = "rf", suffix = suffix, stats_file = stats_file, audience = audience, gram = gram, reducer = reducer)
save_path_log = log_variant_path(save_path)
mkpath(dirname(save_path))
@info "Ensured output directory exists" output_dir = dirname(save_path)

title = sweep_each_experiment ?
    "$(parsed_args["output_dir"]) | Trial $(parsed_args["trial_id"]) | per-step sweep | step=($(every_early),$(every_late)) | $(ncol(train_mat)) programs" :
    "$(parsed_args["output_dir"]) | Trial $(parsed_args["trial_id"]) | depth=$(best_rf_params.max_depth) | split=$(best_rf_params.min_samples_split) | leaf=$(best_rf_params.min_samples_leaf) | feat=$(best_rf_params.max_features) | crit=$(best_rf_params.criterion) | weight=$(best_rf_params.class_weight) | sample=$(best_rf_params.max_samples) | step=($(every_early),$(every_late)) | $(ncol(train_mat)) programs"
best_step_idx = argmax(val_raw)
best_metric_line = has_test ?
    "Best val bacc=$(round(val_raw[best_step_idx], digits = 4)) | test bacc at k=$(xs[best_step_idx]): $(round(test_raw[best_step_idx], digits = 4))" :
    "Best val bacc=$(round(val_raw[best_step_idx], digits = 4)) | test bacc unavailable"
score_line = isempty(stats_file) ?
    "Ranking: feature importance only" :
    "Ranking: $(audience) $(gram) $(reducer) desc, feature importance tie-break"
title = title * "\n" * best_metric_line * "\n" * score_line
interp_bars, interp_labels, interp_bar_scores = isnothing(interpret_scores) ? (Int[], Dict{Int, String}(), Dict{Int, Float64}()) : interpretability_transition_bars(interpret_scores[program_order])
all_bars = sort(unique(vcat(bars, interpretability_bars ? interp_bars : Int[])))
all_bar_labels = Dict{Int, String}()
all_bar_scores = Dict{Int, Float64}()
for (k, v) in interp_labels
    all_bar_labels[k] = v
end
for (k, v) in interp_bar_scores
    all_bar_scores[k] = v
end
plot_front(xs, val_front, val_raw, test_selected, test_raw, save_path, title, baselines; has_test = has_test, model_label = "MAGE+RF", bars = all_bars, bar_labels = all_bar_labels, bar_scores = all_bar_scores, show_bar_labels = show_bar_labels, color_bars = color_bars)
plot_front(xs, val_front, val_raw, test_selected, test_raw, save_path_log, title, baselines; has_test = has_test, model_label = "MAGE+RF", bars = all_bars, bar_labels = all_bar_labels, bar_scores = all_bar_scores, show_bar_labels = show_bar_labels, color_bars = color_bars, xscale = log10)
save_curve_data(save_path, xs, val_front, val_raw, test_selected, test_raw;
    algorithm = "rf",
    trial_id = parsed_args["trial_id"],
    suffix = suffix,
    act = parsed_args["act"],
    audience = audience,
    gram = gram,
    reducer = reducer,
    stats_file = stats_file,
    rank_mode = isempty(stats_file) ? "feature_importance_only" : "interpretability_then_feature_importance",
)

@info "Saved plot to $save_path"
@info "Saved log-x plot to $save_path_log"
