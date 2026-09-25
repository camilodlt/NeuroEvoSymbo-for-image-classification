using ArgParse
using MLJ
using JLD2
using Statistics
using StatsBase: sample
using DataFrames
using MLJ
using CSV
using PythonCall

NeuralNetworkClassifier = MLJ.@load NeuralNetworkClassifier pkg = MLJFlux
DecisionTreeClassifier = MLJ.@load DecisionTreeClassifier pkg = DecisionTree
RandomForestClassifier = MLJ.@load RandomForestClassifier pkg = DecisionTree
LogisticClassifier = MLJ.@load LogisticClassifier pkg = MLJLinearModels
RandomForestClassifierSKLEARN = MLJ.@load RandomForestClassifier pkg = MLJScikitLearnInterface
LogisticClassifierSKLEARN = MLJ.@load LogisticClassifier pkg = MLJScikitLearnInterface
SVC_SKLEARN = MLJ.@load SVMClassifier pkg = MLJScikitLearnInterface

st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))
s = ArgParseSettings()
@add_arg_table s begin
    "--val"
    arg_type = Bool
    default = true
    "--test"
    arg_type = Bool
    default = true
    "--trial_id"
    arg_type = String #
    "--output_dir"
    arg_type = String
    "--boost_round"
    arg_type = Int
    "--use_only_bests"
    arg_type = Bool #
    "--use_really_all"
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
    "--normalize"
    arg_type = Bool
    default = false
    "--lr_max_iter"
    arg_type = Int
    default = 1000
end
rootdir = "./"
Parsed_args = parse_args(s)
@show Parsed_args

BOOST_ROUND = Parsed_args["boost_round"]
USE_BESTS = Parsed_args["use_only_bests"]
USE_REALLY_ALL = Parsed_args["use_really_all"]
HAS_VAL = Parsed_args["val"]
HAS_TEST = Parsed_args["test"]
ACT = Parsed_args["act"]
POOL_FN = Parsed_args["pool_fn"]
NORMALIZE = Parsed_args["normalize"]
LR_MAX_ITER = Parsed_args["lr_max_iter"]

Suffix = "$(USE_BESTS)_$(USE_REALLY_ALL)"
@info "ML data selection" ACT POOL_FN Suffix NORMALIZE

include(joinpath(home, "utils", "datasets.jl"))
include(joinpath(home, "utils", "more_utils.jl"))
include(joinpath(home, "utils", "ml_preprocessing.jl"))

function dataset_file_path(base_dir::String, split::String, suffix::String; act::String, pool_fn::String)
    filename = "best_modules_dataset_$(split)_bests$(suffix).jld2"
    if !isempty(pool_fn)
        return joinpath(base_dir, act, pool_fn, filename)
    end
    act_path = joinpath(base_dir, act, filename)
    legacy_path = joinpath(base_dir, filename)
    if isfile(act_path)
        return act_path
    end
    return legacy_path
end

# JOIN BOOSTING ROUNDS
train_paths = []
val_paths = []
test_paths = []
for round in 0:(BOOST_ROUND)
    p = joinpath(Parsed_args["output_dir"], Parsed_args["trial_id"], "best_modules_boost_$round")
    push!(train_paths, dataset_file_path(p, "train", Suffix; act = ACT, pool_fn = POOL_FN))
    if HAS_VAL
        push!(val_paths, dataset_file_path(p, "val", Suffix; act = ACT, pool_fn = POOL_FN))
    end
    if HAS_TEST
        push!(test_paths, dataset_file_path(p, "test", Suffix; act = ACT, pool_fn = POOL_FN))
    end
end
if Parsed_args["second_trial"] != ""
    for round in 0:(BOOST_ROUND)
        p = joinpath(Parsed_args["output_dir"], Parsed_args["second_trial"], "best_modules_boost_$round")
        push!(train_paths, dataset_file_path(p, "train", Suffix; act = ACT, pool_fn = POOL_FN))
        if HAS_VAL
            push!(val_paths, dataset_file_path(p, "val", Suffix; act = ACT, pool_fn = POOL_FN))
        end
        if HAS_TEST
            push!(test_paths, dataset_file_path(p, "test", Suffix; act = ACT, pool_fn = POOL_FN))
        end
    end
end
@info "Train dataset paths" train_paths
@info "Validation dataset paths" val_paths
@info "Test dataset paths" test_paths

# READ DATA
trainx, trainy = [], []
valx, valy = [], []
testx, testy = [], []
train_mat, val_mat, test_mat = nothing, nothing, nothing

for (x, y, paths) in ((trainx, trainy, train_paths), (valx, valy, val_paths), (testx, testy, test_paths))
    for path in paths
        @info "Reading Data at $path"
        @assert isfile(path) "Dataset file does not exist: $path"
        data = JLD2.load(path)["single_stored_object"]
        push!(x, data.ys)
        push!(y, data.gt)
    end
end

trainx = reduce(hcat, map(x -> reduce(vcat, x'), trainx))
n = length(trainy)
@assert all([all(trainy[1] .== trainy[i]) for i in 1:n])
trainy = trainy[1]

if HAS_VAL
    valx = reduce(hcat, map(x -> reduce(vcat, x'), valx))
    n = length(valy)
    @assert all([all(valy[1] .== valy[i]) for i in 1:n])
    valy = valy[1]
else
    @info "Since no val data, making own val split at 0.8"
    (trainx, trainy), (valx, valy) = MLUtils.splitobs((trainx, trainy); at = 0.8, shuffle = true, stratified = trainy)
end

if HAS_TEST
    testx = reduce(hcat, map(x -> reduce(vcat, x'), testx))
    n = length(testy)
    @assert all([all(testy[1] .== testy[i]) for i in 1:n])
    testy = testy[1]
    test_mat = DataFrame(testx, :auto)
    coerce!(test_mat, Count => Continuous)
    @info "TEST SHAPE : X $(size(testx)) Y $(size(testy))"
end

@info "TRAIN SHAPE : X $(size(trainx)) Y $(size(trainy))"
@info "VAL SHAPE : X $(size(valx)) Y $(size(valy))"

train_mat = DataFrame(trainx, :auto)
val_mat = DataFrame(valx, :auto)
coerce!(train_mat, Count => Continuous)
coerce!(val_mat, Count => Continuous)

# DROP DUPLICATES #
unique_train_data = unique(last, pairs(eachcol(train_mat)))
unique_train_cols = string.(first.(unique_train_data))
dropped_cols = setdiff(names(train_mat), unique_train_cols)
if length(dropped_cols) > 0
    @info "Number of dropped cols $(length(dropped_cols))"
    select!(train_mat, Not(dropped_cols))
    select!(val_mat, Not(dropped_cols))
    select!(test_mat, Not(dropped_cols))
    trainx = Matrix(train_mat)
    valx = Matrix(val_mat)
    testx = Matrix(test_mat)
    @assert size(train_mat) == size(trainx) && size(val_mat) == size(valx) && size(test_mat) == size(testx)
    @info "NEW TRAIN SHAPE : X $(size(train_mat)) Y $(size(trainy))"
    @info "NEW VAL SHAPE : X $(size(val_mat)) Y $(size(valy))"
    @info "NEW TEST SHAPE : X $(size(test_mat)) Y $(size(testy))"
end
# END DROP DUPLICATES #

function correct_ys!(ys::AbstractVector)
    ys_unique = unique(ys)
    n_unique = length(ys_unique)
    has_one = 1.0 in ys_unique
    has_two = 2.0 in ys_unique
    has_three = 3.0 in ys_unique
    if n_unique == 2 && has_one && has_two
        ys[ys .== 1.0] .= -1.0 # 1 to -1
        ys[ys .== 2.0] .= 1.0 # 2 to 1
        # so in the end the vector is {-1,1}
    elseif n_unique == 3 # has 1, 2, 3
        ys .-= 2 # [1, 2, 3] => [-1,0,1]
    else
        @warn "Problem with column that had unique $ys_unique"
    end
    return
end

for df in (train_mat, val_mat, test_mat)
    for c in 1:ncol(df)
        @info "Col $c"
        correct_ys!(@view df[:, c])
    end
end

train_mat, val_mat, test_mat = maybe_standard_scale_prediction_matrices(
    train_mat,
    val_mat,
    test_mat;
    normalize = NORMALIZE,
)

function calculate_acc_bacc_auc(mach, x, y)
    y_hat_prob = predict(mach, x)
    y_hat = predict_mode(mach, x)
    acc = StatisticalMeasures.accuracy(y_hat, y)
    bacc = StatisticalMeasures.balanced_accuracy(y_hat, y)
    # auc = StatisticalMeasures.AreaUnderCurve()(y_hat_prob, y)
    macro_auc, _ = multiclass_auc([i for i in y_hat_prob], y)
    return acc, bacc, macro_auc
end

function tune_lr_params_with_sklearn_grid(train_mat, trainy, val_mat, valy; max_iter::Int)
    sklearn = pyimport("sklearn")
    builtins = pyimport("builtins")
    np = pyimport("numpy")
    model_selection = sklearn.model_selection
    linear_model = sklearn.linear_model

    x_train = Matrix(train_mat)
    x_val = Matrix(val_mat)
    y_train = Int.(trainy)
    y_val = Int.(valy)

    x_all = vcat(x_train, x_val)
    y_all = vcat(y_train, y_val)
    x_all_np = np.asarray(x_all)
    y_all_np = np.asarray(y_all)
    test_fold_np = np.asarray(vcat(fill(-1, size(x_train, 1)), fill(0, size(x_val, 1))))
    ps = model_selection.PredefinedSplit(test_fold_np)

    param_grid = builtins.dict(
        C = [0.001, 0.01, 0.1],
        penalty = ["elasticnet"],
        l1_ratio = [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    base_estimator = linear_model.LogisticRegression(
        class_weight = "balanced",
        max_iter = max_iter,
        solver = "saga",
        random_state = 1,
        n_jobs = -1
    )

    @info "Starting LR GridSearchCV" max_iter n_train = size(x_train, 1) n_val = size(x_val, 1) n_features = size(x_train, 2)
    grid = model_selection.GridSearchCV(
        estimator = base_estimator,
        param_grid = param_grid,
        scoring = "balanced_accuracy",
        cv = ps,
        refit = true,
        n_jobs = -1,
        verbose = 0
    )
    grid.fit(x_all_np, y_all_np)
    best_params = pyconvert(Dict{String, Any}, grid.best_params_)
    best_val_bacc = pyconvert(Float64, grid.best_score_)
    @info "Finished LR GridSearchCV" best_params best_val_bacc max_iter
    return (params = best_params, val_bacc = best_val_bacc)
end

function tune_rf_params_with_sklearn_grid(train_mat, trainy, val_mat, valy)
    sklearn = pyimport("sklearn")
    builtins = pyimport("builtins")
    np = pyimport("numpy")
    model_selection = sklearn.model_selection
    ensemble = sklearn.ensemble

    x_train = Matrix(train_mat)
    x_val = Matrix(val_mat)
    y_train = Int.(trainy)
    y_val = Int.(valy)

    x_all = vcat(x_train, x_val)
    y_all = vcat(y_train, y_val)
    x_all_np = np.asarray(x_all)
    y_all_np = np.asarray(y_all)
    test_fold_np = np.asarray(vcat(fill(-1, size(x_train, 1)), fill(0, size(x_val, 1))))
    ps = model_selection.PredefinedSplit(test_fold_np)

    # Parallelize inside RF (all cores) and keep GridSearchCV single-threaded
    # to avoid nested parallelism.
    param_grid = builtins.dict(
        max_depth = [10, 20, 30],
        min_samples_split = [5, 10, 20],
        max_samples = [0.6, 0.7, 0.8],
        min_samples_leaf = [2, 5],
        max_features = ["sqrt", 0.5],
        criterion = ["gini"]
    )

    base_estimator = ensemble.RandomForestClassifier(
        n_estimators = 100,
        class_weight = "balanced",
        random_state = 1,
        n_jobs = 14
    )

    @info "Starting RF GridSearchCV" n_train = size(x_train, 1) n_val = size(x_val, 1) n_features = size(x_train, 2) n_estimators = 100
    grid = model_selection.GridSearchCV(
        estimator = base_estimator,
        param_grid = param_grid,
        scoring = "balanced_accuracy",
        cv = ps,
        refit = true,
        n_jobs = 1,
        verbose = 0
    )
    grid.fit(x_all_np, y_all_np)
    best_params = pyconvert(Dict{String, Any}, grid.best_params_)
    best_val_bacc = pyconvert(Float64, grid.best_score_)
    @info "Finished RF GridSearchCV" best_params best_val_bacc
    return (params = best_params, val_bacc = best_val_bacc)
end


# SVC ###################################
svc_model = SVC_SKLEARN(
    C = 1.0
)
mach = machine(svc_model, train_mat, categorical(trainy)) |> MLJ.fit!
val_acc = StatisticalMeasures.accuracy(predict(mach, val_mat), valy)
val_bacc = StatisticalMeasures.balanced_accuracy(predict(mach, val_mat), valy)
@info "SVC Val acc : $val_acc. Val BACC : $val_bacc"
test_acc = StatisticalMeasures.accuracy(predict(mach, test_mat), testy)
test_bacc = StatisticalMeasures.balanced_accuracy(predict(mach, test_mat), testy)
@info "SVC TEST acc : $test_acc. TEST BACC : $test_bacc"


sklearn = pyimport("sklearn")
sk_svc = sklearn.svm.SVC(C = 5.0, class_weight = "balanced", kernel = "poly", degree = 3, random_state = 1, cache_size = 4000, max_iter = 10_000)
sk_svc.fit(Matrix(train_mat), trainy .- 1)
val_preds = sk_svc.predict(Matrix(val_mat))
val_preds = pyconvert(Array, val_preds) .+ 1
svc_val_acc = StatisticalMeasures.accuracy(val_preds, valy)
svc_val_bacc = StatisticalMeasures.balanced_accuracy(val_preds, valy)
@info "SVC BALANCED VAL acc : $svc_val_acc. VAL BACC : $svc_val_bacc"

svc_test_acc = missing
svc_test_bacc = missing
if HAS_TEST
    test_preds = sk_svc.predict(Matrix(test_mat))
    test_preds = pyconvert(Array, test_preds) .+ 1
    svc_test_acc = StatisticalMeasures.accuracy(test_preds, testy)
    svc_test_bacc = StatisticalMeasures.balanced_accuracy(test_preds, testy)
    @info "SVC BALANCED TEST acc : $svc_test_acc. TEST BACC : $svc_test_bacc"
end


# RF ###################################
best_rf = tune_rf_params_with_sklearn_grid(train_mat, trainy, val_mat, valy)
rf_best_model = RandomForestClassifierSKLEARN(
    n_estimators = 300,
    max_depth = Int(best_rf.params["max_depth"]),
    class_weight = "balanced",
    criterion = String(best_rf.params["criterion"]),
    min_samples_split = Int(best_rf.params["min_samples_split"]),
    min_samples_leaf = Int(best_rf.params["min_samples_leaf"]),
    max_features = best_rf.params["max_features"],
    max_samples = Float64(best_rf.params["max_samples"]),
    n_jobs = -1,
    random_state = 1
)
rf_best_mach = machine(rf_best_model, train_mat, categorical(trainy)) |> MLJ.fit!
rf_train_acc, rf_train_bacc, rf_train_auc = calculate_acc_bacc_auc(rf_best_mach, train_mat, trainy)
rf_val_acc, rf_val_bacc, rf_val_auc = calculate_acc_bacc_auc(rf_best_mach, val_mat, valy)
@info "RF best model metrics" rf_train_acc rf_train_bacc rf_train_auc rf_val_acc rf_val_bacc rf_val_auc best_rf_params = best_rf.params grid_val_bacc = best_rf.val_bacc
if HAS_TEST
    rf_test_acc, rf_test_bacc, rf_test_auc = calculate_acc_bacc_auc(rf_best_mach, test_mat, testy)
    @info "RF best model test metrics" rf_test_acc rf_test_bacc rf_test_auc
end

RF_METRICS = [(
    val_bacc = rf_val_bacc,
    val_acc = rf_val_acc,
    val_auc = rf_val_auc,
    train_bacc = rf_train_bacc,
    train_acc = rf_train_acc,
    train_auc = rf_train_auc,
    depth = Int(best_rf.params["max_depth"]),
    model = rf_best_mach,
    params = best_rf.params,
)]

# Logistic Regression (tuned with sklearn GridSearchCV on train+val split)
best_lr = tune_lr_params_with_sklearn_grid(train_mat, trainy, val_mat, valy; max_iter = LR_MAX_ITER)
logistic_reg_model = LogisticClassifierSKLEARN(
    class_weight = "balanced",
    max_iter = LR_MAX_ITER,
    solver = "saga",
    l1_ratio = Float64(best_lr.params["l1_ratio"]),
    C = Float64(best_lr.params["C"]),
    penalty = String(best_lr.params["penalty"]),
    random_state = 1,
    n_jobs = -1
)
logistic_machine = machine(logistic_reg_model, train_mat, categorical(trainy)) |> MLJ.fit!
val_acc, val_bacc, val_auc = calculate_acc_bacc_auc(logistic_machine, val_mat, valy)
@info "Logistic Val acc : $(val_acc) . Val Bacc : $(val_bacc) . Val AUC : $(val_auc)"
lr_val_acc = val_acc
lr_val_bacc = val_bacc
if HAS_TEST
    test_acc, test_bacc, test_auc = calculate_acc_bacc_auc(logistic_machine, test_mat, testy)
    @info "Logistic Test acc : $(test_acc) . Test Bacc : $(test_bacc) . Test AUC : $(test_auc)"
    lr_test_acc = test_acc
    lr_test_bacc = test_bacc
else
    lr_test_acc = missing
    lr_test_bacc = missing
end

RF_test = nothing
LR_test = nothing
if HAS_TEST # RUN MODELS ON TEST DATA
    # RUN RANDOM FOREST ---
    best_rf_model = RF_METRICS[1].model
    best_depth = RF_METRICS[1].depth
    @show best_rf_model best_depth

    test_acc, test_bacc, test_auc = calculate_acc_bacc_auc(best_rf_model, test_mat, testy)

    @info "TEST ACC : $test_acc"
    @info "TEST BACC : $test_bacc"
    @info "TEST AUC : $test_auc"

    RF_test = (test_acc = test_acc, test_bacc = test_bacc, test_auc = test_auc)

    # RUN LR ---
    test_acc, test_bacc, test_auc = calculate_acc_bacc_auc(logistic_machine, test_mat, testy)

    @info "TEST ACC : $test_acc"
    @info "TEST BACC : $test_bacc"
    @info "TEST AUC : $test_auc"

    LR_test = (test_acc = test_acc, test_bacc = test_bacc, test_auc = test_auc)
end

@info "Final model summary (val/test)"
@info "SVC val/test" val_acc = svc_val_acc val_bacc = svc_val_bacc test_acc = svc_test_acc test_bacc = svc_test_bacc
@info "RF val/test + params" val_acc = rf_val_acc val_bacc = rf_val_bacc test_acc = (HAS_TEST ? RF_test.test_acc : missing) test_bacc = (HAS_TEST ? RF_test.test_bacc : missing) params = best_rf.params
@info "LR val/test + params" val_acc = lr_val_acc val_bacc = lr_val_bacc test_acc = lr_test_acc test_bacc = lr_test_bacc params = best_lr.params

# NN
# shuffled_indices = shuffle(1:nrow(train_mat))
# train_mat = convert.(Float32, train_mat)
# trainy_cat = categorical(trainy)

# val_mat = convert.(Float32, val_mat)
# valy_cat = categorical(valy)

# clf = NeuralNetworkClassifier(;
#     builder = MLJFlux.MLP(; hidden = (10, 20, 100, 20, 10)), acceleration = CUDALibs(), batch_size = 64, epochs = 1,
#     optimiser = Optimisers.Adam(0.000001)
# )
# mach = machine(clf, train_mat[shuffled_indices, :], trainy_cat[shuffled_indices])
# for i in 1:100
#     fit!(mach, verbosity = 2, force = true)
#     train_preds = predict(mach, train_mat)
#     val_preds = predict(mach, val_mat)
#     training_loss = MLJ.cross_entropy(train_preds, trainy_cat) |> mean
#     train_acc = MLJ.balanced_accuracy(mode.(train_preds), trainy)
#     val_acc = MLJ.balanced_accuracy(mode.(val_preds), valy)
#     @info "Epoch $i : Loss : $training_loss Bacc : $train_acc Val Bacc : $val_acc"
# end
# r = range(clf, :epochs, lower = 1, upper = 200, scale = :log10)
# curve = learning_curve(
#     clf, train_mat, categorical(trainy),
#     range = r,
#     resampling = Holdout(fraction_train = 0.7),
#     measure = MLJ.cross_entropy
# )

# SAVE = ml_models
folder = Parsed_args["output_dir"]
folder = joinpath(folder, Parsed_args["trial_id"], "ml_models_boost_$BOOST_ROUND")
isdir(folder) || mkdir(folder)
# save_object(
#     joinpath(folder, "best_rf_boost_$(BOOST_ROUND).jld2"),
#     RF_METRICS[1]
# )
# save_object(
#     joinpath(folder, "best_dt_boost_$(BOOST_ROUND).jld2"),
#     DT_METRICS[1]
# )

open(joinpath(folder, "best_metrics_bests$(USE_BESTS).txt"), "w") do f
    write(
        f,
        string((val_acc = RF_METRICS[1].val_acc, val_bacc = RF_METRICS[1].val_bacc, val_auc = RF_METRICS[1].val_auc)), "\n"
    )
    write(
        f,
        write(f, string(RF_test), "\n")
    )

    if HAS_TEST
        write(f, string(LR_test), "\n")
    end
end

CSV.write(joinpath(folder, "trainx_bests$(USE_BESTS).csv"), train_mat)
CSV.write(joinpath(folder, "valx_bests$(USE_BESTS).csv"), val_mat)
CSV.write(joinpath(folder, "testx_bests$(USE_BESTS).csv"), test_mat)

CSV.write(joinpath(folder, "trainy_bests$(USE_BESTS).csv"), DataFrame(y = trainy))
CSV.write(joinpath(folder, "valy_bests$(USE_BESTS).csv"), DataFrame(y = valy))
CSV.write(joinpath(folder, "testy_bests$(USE_BESTS).csv"), DataFrame(y = testy))


# #
# using DataStructures
# using JLD2
# P = "BloodMNIST_TPG/EXP_BINARY_STUDENTS_CNN/nn_surrogates_boost_0"
# nn_paths_train = filter(x -> occursin(r"surrogate.*_train", x), readdir(P))
# nn_paths_val = filter(x -> occursin(r"surrogate.*_val", x), readdir(P))
# nn_files_train = Dict()
# nn_files_val = Dict()
# for path in nn_paths_train
#     k = match(r"[0-9]_[0-9]*", path).match
#     K = (parse(Int, k[1]), parse(Int, k[3:end]))
#     nn_files_train[K] = (
#         nn_preds = (JLD2.load(joinpath(P, path))["single_stored_object"]).ys,
#         path = path,
#     )
# end
# for path in nn_paths_val
#     k = match(r"[0-9]_[0-9]*", path).match
#     K = (parse(Int, k[1]), parse(Int, k[3:end]))
#     nn_files_val[K] = (
#         nn_preds = (JLD2.load(joinpath(P, path))["single_stored_object"]).ys,
#         path = path,
#     )
# end

# Keys = sort(collect(keys(nn_files_train)))
# n_keys = length(Keys)
# TRAIN_NN_PREDS = Matrix{Float64}(undef, nrow(train_mat), n_keys)
# VAL_NN_PREDS = Matrix{Float64}(undef, nrow(val_mat), n_keys)
# # TEST_NN_PREDS = Matrix{Float64}(undef, nrow(test_mat), n_keys)

# for key in Keys
#     col = key[2]
#     TRAIN_NN_PREDS[:, col] .= nn_files_train[key].nn_preds
# end

# for key in Keys
#     col = key[2]
#     VAL_NN_PREDS[:, col] .= nn_files_val[key].nn_preds
# end

# sk_svc = sklearn.svm.SVC(C = 5.0, class_weight = "balanced", kernel = "poly", degree = 3, random_state = 1)
# sk_svc.fit(TRAIN_NN_PREDS, trainy .- 1)
# val_preds = sk_svc.predict(VAL_NN_PREDS)
# val_preds = pyconvert(Array, val_preds) .+ 1
# val_acc = StatisticalMeasures.accuracy(val_preds, valy)
# val_bacc = StatisticalMeasures.balanced_accuracy(val_preds, valy)
# @info "SVC BALANCED VAL acc : $val_acc. VAL BACC : $val_bacc"


# # SVC
# # MetaModels = []
# MetaTrain = Matrix{Float64}(undef, nrow(train_mat), n_keys)
# MetaVal = Matrix{Float64}(undef, nrow(val_mat), n_keys)
# MetaTest = Matrix{Float64}(undef, nrow(test_mat), n_keys)

# for c in 1:n_keys
#     @info "Fitting $c"
#     model = RandomForestClassifierSKLEARN(
#         n_estimators = 500, max_depth = 20, class_weight = "balanced",
#         min_samples_split = 5, max_samples = 0.8, n_jobs = -1, random_state = 1
#     )
#     mach = machine(model, train_mat, categorical(TRAIN_NN_PREDS[:, c] .+ 1)) |> MLJ.fit!

#     # sk_svc = sklearn.svm.SVC(C = 5.0, class_weight = "balanced", kernel = "poly", degree = 3, random_state = 1)
#     # sk_svc.fit(Matrix(train_mat), TRAIN_NN_PREDS[:, c])

#     train_preds = predict_mode(mach, Matrix(train_mat))
#     MetaTrain[:, c] .= train_preds #.+ 1

#     val_preds = predict_mode(mach, Matrix(val_mat))
#     MetaVal[:, c] .= val_preds

#     test_preds = predict_mode(mach, Matrix(test_mat))
#     MetaTest[:, c] .= test_preds
# end

# MetaTrain = DataFrame(MetaTrain, :auto)
# MetaVal = DataFrame(MetaVal, :auto)
# MetaTest = DataFrame(MetaTest, :auto)
# coerce!(MetaTrain, Count => Continuous)
# coerce!(MetaVal, Count => Continuous)
# coerce!(MetaTest, Count => Continuous)

# model = RandomForestClassifierSKLEARN(
#     n_estimators = 500, max_depth = 20, class_weight = "balanced",
#     min_samples_split = 5, max_samples = 0.8, n_jobs = -1, random_state = 1
# )
# mach = machine(model, MetaTrain, categorical(trainy)) |> MLJ.fit!
# val_preds = predict_mode(mach, MetaVal)
# val_acc = StatisticalMeasures.accuracy(val_preds, valy)
# val_bacc = StatisticalMeasures.balanced_accuracy(val_preds, valy)
# @info "SVC BALANCED VAL acc : $val_acc. VAL BACC : $val_bacc"

# test_preds = predict_mode(mach, MetaTest)
# test_acc = StatisticalMeasures.accuracy(test_preds, testy)
# test_bacc = StatisticalMeasures.balanced_accuracy(test_preds, testy)
# @info "SVC BALANCED test acc : $test_acc. test BACC : $test_bacc"
