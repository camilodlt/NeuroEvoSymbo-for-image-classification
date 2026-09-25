using ArgParse
st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(file))
s = ArgParseSettings()
@add_arg_table s begin
    "--data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    "--val_data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--test_data_location"
    arg_type = String # where to read the data to make a features dataset by running the models
    default = ""
    "--trial_id"
    arg_type = String #
    "--all"
    arg_type = Bool #
    default = true
    "--output_dir"
    arg_type = String
    "--use_ski"
    arg_type = Bool
    default = false
    "--act"
    arg_type = String
    default = "identity"
    "--use_new_number_extensions"
    arg_type = Bool
    default = false
    "--use_new_intensityimg_extensions"
    arg_type = Bool
    default = false
    "--use_new_binaryimg_extensions"
    arg_type = Bool
    default = false
    "--use_new_segmentimg_extensions"
    arg_type = Bool
    default = false
    "--use_imagegraph_bundle"
    arg_type = Bool
    default = false
end
rootdir = "./"
Parsed_args = parse_args(s)
@show Parsed_args

import MLJ
include(joinpath(home, "src", "mage_imports.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))

USE_ALL = Parsed_args["all"]
USE_SKI = Parsed_args["use_ski"]
USE_SKI ? addprocs(nt, exeflags = ["--threads=1"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images

include(joinpath(home, "utils", "utils.jl"))
include(joinpath(home, "utils", "utils_aml.jl"))
include(joinpath(home, "utils", "datasets.jl"))
include(joinpath(home, "utils", "activations.jl"))
include(joinpath(home, "utils", "more_utils.jl"))


use_best = Parsed_args["all"]

# Read data ---
data_path = Parsed_args["data_location"]
data = JLD2.load(data_path)["single_stored_object"]
trainx, trainy = data.xs, data.ys
trainx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in trainx]

val_location = Parsed_args["val_data_location"]
test_location = Parsed_args["test_data_location"]
has_val_data = val_location != ""
has_test_data = test_location != ""
if has_val_data
    data = JLD2.load(val_location)["single_stored_object"]
    valx, valy = data.xs, data.ys
    valx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in valx]
    @assert length(valx) == length(valy)
else
    valx, valy = nothing, nothing
    VALDataloader = nothing
end

testx, testy = nothing, nothing
if has_test_data
    data = JLD2.load(test_location)["single_stored_object"]
    testx, testy = data.xs, data.ys
    testx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in testx]
    @assert length(testx) == length(testy)
end

@assert length(trainx) == length(trainy)
@info "Size train : $(length(trainx))"
@info "Size val : $(length(valx))"

const CLASSES = sort(unique(trainy))
const N_CLASSES = length(CLASSES)
const sample_img = trainx[1][1]

include(joinpath(home, "src", "magenet_image_bundles.jl"))
define_common_image_functions(sample_img)
skimage_factories = USE_SKI ? setup_skimage_distributed(Type2Dimg_binary) : nothing

# Float Bundles
float_bundles = UTCGP.get_float_bundles()
USE_SKI ? push!(float_bundles, skimage_factories...) : nothing
add_requested_float_extension_bundles!(float_bundles; parsed_args = Parsed_args)
glcm_b = deepcopy(UTCGP.experimental_bundle_float_glcm_factory)
push!(float_bundles, glcm_b)

set_bundle_casters!(float_bundles, float_caster2)

# Metalibs
ml = ml_from_vbundles([image_intensity, image_binary, image_segment, float_bundles])
n_ins = length(trainx[1])

model_arch = modelArchitecture( # TODO
    [Type2Dimg_intensity for i in 1:n_ins],
    [1 for i in 1:n_ins],
    [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment, Float64],
    [Float64],
    [4]
)

folder = Parsed_args["output_dir"]
folder = joinpath(folder, Parsed_args["trial_id"])
folder = joinpath(folder, "mage_imgcls")
folders_to_read = joinpath.(folder, readdir(folder))
@show folders_to_read

Fitnesses = OrderedDict()
for folder_to_read in folders_to_read
    @info "Reading $folder_to_read"
    for (root, dirs, files) in walkdir(folder_to_read)
        if isempty(files)
            @show root
            continue
        end
        @show files
        metrics_file = filter(x -> occursin("metrics", x), files)[1]
        metrics_file_content = readlines(joinpath(root, metrics_file))
        @info "Reading : $metrics_file"
        try
            metrics = JSON.parse(metrics_file_content[end])
            best_f = metrics["params"]["best_tracker_loss"]
            Fitnesses[root] = (loss = best_f, root = root, metrics_file = metrics_file, ind_path = joinpath(root, "checkpoint_0.pickle"))
        catch e
            @show e
            @warn root
        end
    end
end

best_metric = minimum([v.loss for (k, v) in Fitnesses])
best_ind = minimum([k for (k, v) in Fitnesses if v.loss == best_metric])

Pop = OrderedDict()
Bests = OrderedDict()
for (k, v) in Fitnesses
    Pop[k] = deserialize(v.ind_path)["best_genome"]
    k == best_ind ? Bests[k] = Pop[k] : nothing
end

N_NODES = Bests[best_ind][1] |> length
node_config = nodeConfig(N_NODES, 1, 3, n_ins)
shared_in, _ = make_evolvable_utgenome(
    model_arch, ml, node_config
)

# TRAIN DATA
nt = Threads.nthreads()
UTCGP.reset_genome!.(values(Pop))

# Store all outs
OUTS_train = OrderedDict()
OUTS_val = OrderedDict()
OUTS_test = OrderedDict()

data = []
push!(data, (trainx, trainy, OUTS_train, "train"))
if has_val_data
    push!(data, (valx, valy, OUTS_val, "val"))
end
if has_test_data
    push!(data, (testx, testy, OUTS_test, "test"))
end

Inds = USE_ALL ? Pop : Bests
for (i, kv) in enumerate(Inds)
    k, v = kv
    prog = UTCGP.decode_with_output_nodes(v, ml, model_arch, shared_in)
    progs = [deepcopy(prog) for i in 1:nt]
    pop_size = length(progs)
    @info "Running $k"
    for (datax, datay, store, split) in data
        xs = datax
        ys = datay
        n_samples = length(xs)
        thread_size = ceil(Int, n_samples / pop_size)
        pop_subsets = Iterators.partition(1:n_samples, thread_size)
        OUTS = Vector{NamedTuple}(undef, n_samples)
        tasks = []
        for (idx, pop_subset) in enumerate(pop_subsets)
            t = Threads.@spawn begin
                xs_v, ys_v = xs[pop_subset], ys[pop_subset]
                out_v = @view OUTS[pop_subset]
                prog_copy = deepcopy(progs[idx])
                @assert length(out_v) == length(xs_v) == length(ys_v)
                for (sample_idx, (x, y)) in enumerate(zip(xs_v, ys_v))
                    UTCGP.reset_program!.(prog_copy)
                    UTCGP.replace_shared_inputs!(prog_copy, x)
                    outputs = UTCGP.evaluate_individual_programs(prog_copy, model_arch.chromosomes_types, ml)
                    final_out = INTER_ACT[](outputs)
                    out_v[sample_idx] = (raw_pred = outputs, pred = argmax(final_out), gt = y)
                end
            end
            push!(tasks, t)
        end
        fetch.(tasks)
        store[k] = deepcopy(OUTS)
    end
end

# CALC ALL METRICS INDIVIDUALLY
levels_ = 1:N_CLASSES |> collect |> MLJ.categorical
Metrics_per_individual_train = OrderedDict()
Metrics_per_individual_val = OrderedDict()
Metrics_per_individual_test = OrderedDict()

for (k, all_rows) in OUTS_train
    gt = map(row -> row.gt, all_rows)
    yhat = map(row -> row.pred, all_rows)
    yhat_prob = map(row -> UnivariateFinite(levels_, softmax(row.raw_pred)), all_rows)
    acc = StatisticalMeasures.accuracy(yhat, gt)
    bacc = StatisticalMeasures.balanced_accuracy(yhat, gt)
    macro_auc, _ = multiclass_auc(yhat_prob, gt)
    Metrics_per_individual_train[k] = (acc = acc, bacc = bacc, auc = macro_auc)
end

if has_val_data
    for (k, all_rows) in OUTS_val
        gt = map(row -> row.gt, all_rows)
        yhat = map(row -> row.pred, all_rows)
        yhat_prob = map(row -> UnivariateFinite(levels_, softmax(row.raw_pred)), all_rows)
        acc = StatisticalMeasures.accuracy(yhat, gt)
        bacc = StatisticalMeasures.balanced_accuracy(yhat, gt)
        macro_auc, _ = multiclass_auc(yhat_prob, gt)
        Metrics_per_individual_val[k] = (acc = acc, bacc = bacc, auc = macro_auc)
    end
    open(joinpath(folder, "metrics_per_individual_val.txt"), "w") do io
        for (k, v) in Metrics_per_individual_val
            @show k
            write(io, "$k $(v.acc) $(v.bacc) $(v.auc)", "\n")
        end
    end
end

if has_test_data
    for (k, all_rows) in OUTS_test
        gt = map(row -> row.gt, all_rows)
        yhat = map(row -> row.pred, all_rows)
        yhat_prob = map(row -> UnivariateFinite(levels_, softmax(row.raw_pred)), all_rows)
        acc = StatisticalMeasures.accuracy(yhat, gt)
        bacc = StatisticalMeasures.balanced_accuracy(yhat, gt)
        macro_auc, _ = multiclass_auc(yhat_prob, gt)
        Metrics_per_individual_test[k] = (acc = acc, bacc = bacc, auc = macro_auc)
    end
    open(joinpath(folder, "metrics_per_individual_test.txt"), "w") do io
        for (k, v) in Metrics_per_individual_test
            @show k
            write(io, "$k $(v.acc) $(v.bacc) $(v.auc)", "\n")
        end
    end
end

open(joinpath(folder, "metrics_per_individual_train.txt"), "w") do io
    for (k, v) in Metrics_per_individual_train
        @show k
        write(io, "$k $(v.acc) $(v.bacc) $(v.auc)", "\n")
    end
end

# ENSEMBLE ?
function ensemble_preds(rows, n_classes)
    ks = keys(rows) |> collect
    n = length(rows[ks[1]])
    n_keys = length(ks)
    outs = []
    tmp_mat = zeros(Float64, n_keys, n_classes)
    for i in 1:n
        for (ith_ind, k) in enumerate(ks)
            tmp_mat[ith_ind, :] = softmax(rows[k][i].raw_pred)
        end
        ensemble_pred = mean(tmp_mat, dims = 1)[1, :]
        push!(outs, UnivariateFinite(levels_, ensemble_pred))
    end
    return identity.(outs)
end

# TRAIN ENSEMBLE METRICS
ensemble_probs = ensemble_preds(OUTS_train, N_CLASSES)
ensemble_mode = map(i -> Int(mode(i).ref), ensemble_probs)
ensemble_acc = StatisticalMeasures.accuracy(ensemble_mode, trainy)
ensemble_bacc = StatisticalMeasures.balanced_accuracy(ensemble_mode, trainy)
ensemble_auc, _ = multiclass_auc(ensemble_probs, trainy)
open(joinpath(folder, "ensemble.txt"), "a") do io
    write(io, string((train_acc = ensemble_acc, train_bacc = ensemble_bacc, train_auc = ensemble_auc)), "\n")
end

if has_val_data
    ensemble_probs = ensemble_preds(OUTS_val, N_CLASSES)
    ensemble_mode = map(i -> Int(mode(i).ref), ensemble_probs)
    ensemble_acc = StatisticalMeasures.accuracy(ensemble_mode, valy)
    ensemble_bacc = StatisticalMeasures.balanced_accuracy(ensemble_mode, valy)
    ensemble_auc, _ = multiclass_auc(ensemble_probs, valy)
    open(joinpath(folder, "ensemble.txt"), "a") do io
        write(io, string((val_acc = ensemble_acc, val_bacc = ensemble_bacc, val_auc = ensemble_auc)), "\n")
    end
end

if has_test_data
    ensemble_probs = ensemble_preds(OUTS_test, N_CLASSES)
    ensemble_mode = map(i -> Int(mode(i).ref), ensemble_probs)
    ensemble_acc = StatisticalMeasures.accuracy(ensemble_mode, testy)
    ensemble_bacc = StatisticalMeasures.balanced_accuracy(ensemble_mode, testy)
    ensemble_auc, _ = multiclass_auc(ensemble_probs, testy)
    open(joinpath(folder, "ensemble.txt"), "a") do io
        write(io, string((test_acc = ensemble_acc, test_bacc = ensemble_bacc, test_auc = ensemble_auc)), "\n")
    end
end
