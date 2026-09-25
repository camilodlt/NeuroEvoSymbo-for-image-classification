using CUDA, cuDNN
using PartialFunctions
using PythonCall
import MLJ
using UnicodePlots
using Statistics
# import DataFrames
import MLJ

st = time()
dir = @__DIR__
pwd_dir = pwd()
file = @__FILE__
home = dirname(dirname(dirname(file)))

include(joinpath(home, "src", "magenet_imports.jl"))
include(joinpath(home, "src", "magenet_args.jl"))
include(joinpath(home, "src", "distill", "utils.jl"))
include(joinpath(home, "src", "distill", "distill_argparse.jl"))
Parsed_args = @isdefined(DISTILL_EXTERNAL_PARSED_ARGS) ? DISTILL_EXTERNAL_PARSED_ARGS : parse_distill_args(default_phase = "all")
@show Parsed_args

#################### REMOVE ################
using Infiltrator

MIX_PROB = Parsed_args["mix_prob"] # [0,1]
@assert MIX_PROB >= 0.0 && MIX_PROB <= 1.0
LAMBDA_STUDENT = Parsed_args["lambda_student"] # [0,∞[
MIN_STUDENT_LOSS = Parsed_args["min_student_loss"] # [0,1]
MIN_ENTROPY_LOSS = Parsed_args["min_entropy_loss"] # [0,1]
WEIGHT_ENTROPY = Parsed_args["weight_entropy"] # [0,1]
IS_PARAM_TUNING = Parsed_args["param_tuning"]

TRAIN_BATCH_SIZE = Parsed_args["train_bs"]
VAL_BATCH_SIZE = Parsed_args["val_bs"]
DROPOUT = Parsed_args["dropout"]
WEIGHT_DECAY = Parsed_args["wd"]

BINARIZE = Parsed_args["binarize"]
OPT = Parsed_args["optim"]
LAST_LAYER_TYPE = Symbol(Parsed_args["last_layer_type"])
WORKERS = Parsed_args["workers"]
PRETRAINED_MODEL = Parsed_args["pretrained"] # "vit", "resnet", "false"
TAIL_ACTIVATION = lowercase(Parsed_args["tail_activation"])
TORCHVISION_WEIGHTS = Parsed_args["torchvision_weights"]
DISTILL_PHASE = lowercase(Parsed_args["distill_phase"])
USE_IMAGENET_STATS = Parsed_args["use_imagenet_stats"]
RESIZE_TO = Parsed_args["resize_to"] # img resize, eg 224

NUM_STUDENT = Parsed_args["num_student_type"] # without_student || uni_student || multi_student
STUDENT_ARCH = Symbol(Parsed_args["student_arch"]) # custom cnn || squeezenets
STUDENT_INNER_CHANNELS = Parsed_args["student_inner_channels"] # how many filters in conv
STUDENT_N_MLP = Parsed_args["student_n_mlp"] # how many mlps after flatten
STUDENT_DROPOUT = Parsed_args["student_dropout"] # student dropout after flatten

#################### END REMOVE ################

LIP = Parsed_args["lip"]
LR = Parsed_args["lr"]
SCHEDULE = Parsed_args["schedule"]
BOOST_ROUND = Parsed_args["boost_round"]
@info "BOOSTING ROUND $BOOST_ROUND"
include(joinpath(home, "src", "magenet_ski.jl"))

"""Append one CLI arg from parsed args if present and non-`nothing`."""
function _push_distill_cli_arg!(cli_args::Vector{String}, parsed_args, key::String)
    haskey(parsed_args, key) || return
    value = parsed_args[key]
    value === nothing && return
    push!(cli_args, "--$(key)=$(value)")
    return
end

"""Build explicit forwarded CLI args from `Parsed_args` and override only `distill_phase`."""
function distill_cli_args(parsed_args, phase::String)
    ordered_keys = (
        "seed", "data_location", "val_data_location", "test_data_location",
        "output_dir", "trial_id",
        "mutation_rate", "mutation_n_models", "n_nodes", "n_new", "n_elite", "tour_size",
        "n_samples", "n_repetitions", "err_w", "time_w", "time",
        "generations_mage", "lambda_mage", "trainsize_mage",
        "device", "epochs", "n_surr", "n_nn_batches_train", "n_nn_batches_val",
        "n_layers", "n_layers_img", "l1_size", "l1_out_size", "l2_size", "l2_out_size", "l3_size", "l4_size",
        "th_for_align", "th_for_es", "reset_pb",
        "dropout_rate", "freeze_rate", "batch_size", "regularization", "inter_losses",
        "mask", "optim", "max", "use_ski", "gens",
        "boost_round", "latent_dim", "lip", "param_tuning",
        "val_bs", "workers", "pretrained", "tail_activation", "torchvision_weights",
        "use_imagenet_stats", "resize_to", "train_bs", "dropout", "last_layer_type",
        "lr", "wd", "schedule", "mix_prob", "min_entropy_loss", "weight_entropy",
        "num_student_type", "student_arch", "student_inner_channels", "student_n_mlp", "student_dropout",
        "lambda_student", "min_student_loss", "binarize",
    )

    cli_args = String[]
    for key in ordered_keys
        _push_distill_cli_arg!(cli_args, parsed_args, key)
    end
    push!(cli_args, "--distill_phase=$(phase)")
    return cli_args
end

"""Run a distillation phase script in a subprocess while forwarding parsed CLI arguments."""
function run_distill_phase_subprocess!(script_path::String, phase::String)
    if get(Parsed_args, "data_location", "") == ""
        error("Parsed_args[\"data_location\"] is empty before subprocess launch. Check caller CLI argument formatting.")
    end
    project_dir = dirname(Base.active_project())
    phase_args = distill_cli_args(Parsed_args, phase)
    cmd = `$(Base.julia_cmd()) --project=$project_dir -t $(Threads.nthreads()) $script_path $phase_args`
    @warn "Launching distill subprocess" phase forwarded_args_count = length(phase_args) command = string(cmd)
    run(cmd)
    return
end

if DISTILL_PHASE == "all"
    run_distill_phase_subprocess!(joinpath(home, "src", "distill", "train_resnet18_latent_train.jl"), "train_only")
    run_distill_phase_subprocess!(joinpath(home, "src", "distill", "train_resnet18_latent_capture.jl"), "capture_only")
    exit(0)
elseif DISTILL_PHASE == "train_only"
    run_distill_phase_subprocess!(joinpath(home, "src", "distill", "train_resnet18_latent_train.jl"), "train_only")
    exit(0)
elseif DISTILL_PHASE == "capture_only"
    run_distill_phase_subprocess!(joinpath(home, "src", "distill", "train_resnet18_latent_capture.jl"), "capture_only")
    exit(0)
else
    error("Unsupported --distill_phase=$DISTILL_PHASE. Use all, train_only, or capture_only.")
end

const_vars = setup_constants(Parsed_args)
map(i -> eval(i), const_vars)
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images

# MAGENetwork.λ_cosinesim[] = REGULARIZATION # FIX
MAGENetwork.DP_RATE[] = DROPOUT #DROPOUT_RATE # FIX
MAGENetwork.OPTIM[] = OPTIM # FIX

include_utils_and_disable_logging(home)

Parsed_args["type_of_module"] = "v1"
type_of_module = Symbol(Parsed_args["type_of_module"])

"""Map a CLI tail-activation name to the matching Julia activation function."""
function resolve_tail_activation(name::String)
    if name in ("identity", "none", "linear")
        return identity
    elseif name == "sigmoid"
        return sigmoid
    elseif name == "relu"
        return relu
    elseif name == "tanh"
        return tanh
    elseif name == "asinh"
        return asinh
    end
    error("Unsupported --tail_activation=$name. Use identity, sigmoid, relu, tanh, or asinh.")
end

"""Map a CLI tail-activation name to the matching Torch activation module."""
function resolve_tail_activation(pybackend, name::String)
    if name in ("identity", "none", "linear")
        return pybackend.torch.nn.Identity()
    elseif name == "sigmoid"
        return pybackend.torch.nn.Sigmoid()
    elseif name == "relu"
        return pybackend.torch.nn.ReLU()
    elseif name == "tanh"
        return pybackend.torch.nn.Tanh()
    elseif name == "asinh"
        return @pyeval (torch = pybackend.torch,) => "type('AsinhActivation', (torch.nn.Module,), {'forward': lambda self, x: torch.asinh(x)})()"
    end
    error("Unsupported --tail_activation=$name. Use identity, sigmoid, relu, tanh, or asinh.")
end

"""Normalize torchvision weights CLI input into a value accepted by torchvision builders."""
function resolve_torchvision_weights(name::String)
    normalized = lowercase(name)
    if normalized in ("none", "nothing", "false", "random")
        return nothing
    elseif normalized == "default"
        return "DEFAULT"
    end
    error("Unsupported --torchvision_weights=$name. Use DEFAULT or none.")
end

"""Persist the best tail/head PyTorch weights produced during distillation training."""
function save_distill_weights!(outs, folder)
    @pyeval (w = outs[2].best_weights[1], filename = joinpath(folder, "model_weights_tail.pt")) => "torch.save(w, filename)"
    @pyeval (w = outs[2].best_weights[2], filename = joinpath(folder, "model_weights_head.pt")) => "torch.save(w, filename)"
    return
end

"""Load previously saved tail/head PyTorch weights from disk into the current models."""
function load_distill_weights!(pybackend, model_tail, model_head, folder)
    tail_file = joinpath(folder, "model_weights_tail.pt")
    head_file = joinpath(folder, "model_weights_head.pt")
    @pyexec (
        m = model_tail.nn,
        torch = pybackend.torch,
        filename = tail_file,
        loc = "cpu",
    ) => "m.load_state_dict(torch.load(filename, map_location = loc))"
    @pyexec (
        m = model_head.nn,
        torch = pybackend.torch,
        filename = head_file,
        loc = "cpu",
    ) => "m.load_state_dict(torch.load(filename, map_location = loc))"
    return
end

# ACTIVATION
MAGENetwork.MODULE_ACT[] = resolve_tail_activation(TAIL_ACTIVATION) # final act for features before head
MAGENetwork.INTER_MODULE_ACT[] = relu #MAGENetwork.groupsort2 # mocks, pylipext will give groupsort2
@info "Tail activation" tail_activation = TAIL_ACTIVATION module_act = MAGENetwork.MODULE_ACT[]
torchvision_weights = resolve_torchvision_weights(TORCHVISION_WEIGHTS)
@info "Torchvision weights" torchvision_weights

EPOCHS = Parsed_args["epochs"]
@show EPOCHS

######################
# READ FILES #########
######################
split_data = load_distill_image_splits(Parsed_args)
data_path = split_data.data_path
val_data_path = split_data.val_data_path
test_data_path = split_data.test_data_path
trainx, trainy = split_data.trainx, split_data.trainy
valx, valy = split_data.valx, split_data.valy
testx, testy = split_data.testx, split_data.testy

const CLASSES = sort(unique(trainy))
const N_CLASSES = length(CLASSES)
const sample_img = trainx[1][1]

# DEPENDS ON THE INPUT SIZE
n_pixels = size(sample_img, 1) * size(sample_img, 2)
BS_TRAIN = TRAIN_BATCH_SIZE
BS_VAL = VAL_BATCH_SIZE
@info "BS : $BS_TRAIN $BS_VAL"

# Image Bundles
include(joinpath(home, "src", "magenet_image_bundles.jl"))
define_common_image_functions(sample_img)
N_INS = trainx[1] |> length

# Float Bundles
float_bundles = UTCGP.get_float_bundles()

# Symbolic Regression Bundles
only_float_bundles = UTCGP.get_sr_float_bundles()

set_bundle_casters!(float_bundles, float_caster2)
set_bundle_casters!(only_float_bundles, float_caster2)

# Metalibs
ml = ml_from_vbundles([image_intensity, image_binary, image_segment, float_bundles])
ml_float = ml_from_vbundles([only_float_bundles])

# PYTORCHLIP BACKEND
_pylipbackend = MAGENetwork.get_pytorchlip_backend()
_pybackend = MAGENetwork.get_pytorch_backend()
pybackend = LIP ? _pylipbackend : _pybackend
torch_cuda_available = pyconvert(Bool, pybackend.torch.cuda.is_available())
torch_cuda_device_count = pyconvert(Int, pybackend.torch.cuda.device_count())
torch_cuda_current_device = torch_cuda_available ? pyconvert(Int, pybackend.torch.cuda.current_device()) : -1
@info "Torch CUDA status" available = torch_cuda_available device_count = torch_cuda_device_count current_device = torch_cuda_current_device
@info "Setting TORCH seed to $SEED"
pybackend.torch.manual_seed(SEED)

# Make new pop
initial_pop = create_initial_magenet_population(
    pybackend,
    1,
    Parsed_args, Type2Dimg_intensity, (Type2Dimg_binary, Type2Dimg_segment), ml, ml_float, valx, 20, N_CLASSES; type_of_module,
    use_surrogate = false
) # 20 was n_nodes

means, stds = stack_and_stats(map(i -> reduce((x, y) -> cat(x, y, dims = 3), map(x -> reinterpret(x.img), i)), trainx))

########################################################################################
####################################### DATA AUG #######################################
########################################################################################
transform_setup = build_distill_transforms(
    sample_img,
    N_INS;
    pretrained_model = PRETRAINED_MODEL,
    use_imagenet_stats = USE_IMAGENET_STATS,
    resize_to = RESIZE_TO,
    means = means,
    stds = stds,
)
means, stds = transform_setup.means, transform_setup.stds
Img_size = transform_setup.img_size
RESIZE_TO = transform_setup.resize_to
py_train_transforms = transform_setup.py_train_transforms
py_val_transforms = transform_setup.py_val_transforms

########################################################################################
################################# TAIL &&  HEADS #######################################
########################################################################################
m = initial_pop[1]
L1 = Parsed_args["l1_size"]
L1_OUT = Parsed_args["l1_out_size"]
L1_OUT = BINARIZE ? L1_OUT : 1 # if it's not binary, l1 out must be one since there is no reshaping

NNEURONS_TOTAL = L1 * L1_OUT
@info "NN neurons : $(NNEURONS_TOTAL) : L1 $L1 and L1_OUT $L1_OUT. It will be reshaped to ($L1, $L1_OUT)"
# MODEL ARCH FOR TAIL
ma_tail = modelArchitecture(
    [[Type2Dimg_intensity for _ in 1:N_INS]...],
    [[1 for _ in 1:N_INS]...],
    [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment, Float64],
    [Float64 for i in 1:NNEURONS_TOTAL], # L1 outputs
    [4 for i in 1:NNEURONS_TOTAL] # L1 outputs
)
model_args_tail = MAGENetwork._args_needed_per_model(MAGENetwork.ImagesToScalarNN, ma_tail, pybackend)

# MODEL ARCH FOR HEAD
ma_head = modelArchitecture(
    [Float64 for _ in 1:L1],
    [1 for _ in 1:L1],
    [Float64],
    [Float64 for _ in 1:N_CLASSES],
    [1 for _ in 1:N_CLASSES]
)
model_args_head = MAGENetwork._args_needed_per_model(MAGENetwork.ScalarsToScalarNN, ma_head, pybackend)

# MAKE TAIL AND HEAD
MODEL_SIZE = :gumbel_softmax
@info "Model size type : $MODEL_SIZE"

pylipext = Base.get_extension(MAGENetwork, :PytorchLip)
tail_activation = resolve_tail_activation(pybackend, TAIL_ACTIVATION)
if PRETRAINED_MODEL == "resnext50_32x4d"
    _model = pylipext.get_resnext50_32x4d()(
        new_hidden_dim = NNEURONS_TOTAL, weights = torchvision_weights, freeze_backbone = false, dropout_mlp = MAGENetwork.DP_RATE[], mlp_hidden_layers = [256, 128], n_channels_in = N_INS,
        activation = tail_activation
    )
    model_tail = MAGENetwork.ImagesToScalarNN(_model, model_args_tail..., :notdefined, false, pybackend)
elseif occursin("resnet", PRETRAINED_MODEL)
    _model = pylipext.get_resnet()(
        new_hidden_dim = NNEURONS_TOTAL, backbone_name = PRETRAINED_MODEL, weights = torchvision_weights, freeze_backbone = false, dropout_mlp = MAGENetwork.DP_RATE[], mlp_hidden_layers = [256, 128], n_channels_in = N_INS,
        activation = tail_activation
    )
    model_tail = MAGENetwork.ImagesToScalarNN(_model, model_args_tail..., :notdefined, false, pybackend)
elseif PRETRAINED_MODEL == "vit"
    _model = pylipext.get_vit()(new_hidden_dim = NNEURONS_TOTAL, backbone_name = "vit_b_16", weights = "DEFAULT", freeze_backbone = false, dropout_mlp = MAGENetwork.DP_RATE[], mlp_hidden_layers = [192], n_channels_in = N_INS, activation = tail_activation)
    model_tail = MAGENetwork.ImagesToScalarNN(_model, model_args_tail..., :notdefined, false, pybackend)
else
    model_tail = MAGENetwork.create_nn_model(pybackend, MAGENetwork.ImagesToScalarNN, model_args_tail...; size = MODEL_SIZE, last = false)
end
model_head = MAGENetwork.create_nn_model(pybackend, MAGENetwork.ScalarsToScalarNN, model_args_head...; size = LAST_LAYER_TYPE, last = true)

# FOLDER
# MAKE FOLDERS TO SAVE
id = Parsed_args["trial_id"]
folder = ensure_distill_output_folder(Parsed_args["output_dir"], id, BOOST_ROUND)

##################################################################################
################################# DATASETS #######################################
##################################################################################
vec_of_3D_imgs_train = nothing
vec_of_3D_imgs_val = nothing
vec_of_3D_imgs_test = nothing
TrainDataset = nothing
ValDataset = nothing
TestDataset = nothing
TrainDataLoader = nothing
ValDataLoader = nothing
TestDataLoader = nothing

pybackend.torch.backends.cudnn.benchmark = false

if DISTILL_PHASE == "train_only"
    dataloader_setup = build_distill_dataloaders(
        pylipext,
        trainx, trainy,
        valx, valy,
        testx, testy,
        py_train_transforms,
        py_val_transforms,
        BS_TRAIN,
        BS_VAL;
        workers = WORKERS,
    )
    TrainDataset = dataloader_setup.train_dataset
    ValDataset = dataloader_setup.val_dataset
    TestDataset = dataloader_setup.test_dataset
    TrainDataLoader = dataloader_setup.train_dataloader
    ValDataLoader = dataloader_setup.val_dataloader
    TestDataLoader = dataloader_setup.test_dataloader

    torchsummary = pyimport("torchsummary")
    torchsummary.summary(model_tail.nn, (N_INS, Img_size, Img_size), depth = 5);
    torchsummary.summary(model_head.nn, (L1,), depth = 3);

    best_lr = LR
    @show best_lr
    MAGENetwork.OPTIM[] = OPT
    MAGENetwork.LR[] = best_lr

    @pyeval (x = ValDataset,) => "x[1]"
    @pyeval (x = TrainDataset,) => "x[1]"
    @pyeval (x = TestDataset,) => "x[1]"

    pybackend.torch.set_float32_matmul_precision("high")
    USE_STUDENTS = pylipext.WithoutStudent()

    outs = try
        MAGENetwork.train_tail_and_heads(
            _pylipbackend, m,
            model_tail, model_head;
            binarize = BINARIZE,
            n_classes = N_CLASSES,
            max_epochs = EPOCHS,
            train_dl = TrainDataLoader,
            val_dl = ValDataLoader,
            to_schedule = SCHEDULE,
            to_penalise_sim = false,

            img_size = Img_size,
            mix_prob = MIX_PROB,
            activation_fn = nothing,
            lambda_cls = 1.0,
            label_smoothing_factor = 0.0,
            alpha_decode = 0.0,
            weight_decay = WEIGHT_DECAY,
            neurons_class = L1,
            neurons_per_class = L1_OUT,
            n_channels = N_INS,
            use_students = USE_STUDENTS,
            student_arch = STUDENT_ARCH,
            student_inner_channels_cnn = STUDENT_INNER_CHANNELS,
            student_n_mlp = STUDENT_N_MLP,
            student_dropout = STUDENT_DROPOUT,
            student_agg = :sum,
            lambda_student = LAMBDA_STUDENT,
            min_student_loss = MIN_STUDENT_LOSS,
            use_student_entropy = true,
            min_entropy_loss = MIN_ENTROPY_LOSS,
            weight_entropy = WEIGHT_ENTROPY,
            use_grads_on_latent_space = false,
            lambda_grad = 0.01,
            grad_eps = 1.0e-2,
            save_best_bacc = true,
            record_all_weights = false,
            record_every = 200,
            unfreeze_at = -1
        )
    catch e
        @show e
        if WORKERS
            try
                TrainDataLoader._iterator._shutdown_workers()
            catch e
                @show e
                @info "Could not shutdown workers ? "
            end
            try
                ValDataLoader._iterator._shutdown_workers()
            catch e
                @show e
                @info "Could not shutdown workers ? "
            end
        end
        PythonCall.pydel!(TrainDataset)
        PythonCall.pydel!(ValDataset)
        PythonCall.pydel!(TestDataset)
        PythonCall.pydel!(TrainDataLoader)
        PythonCall.pydel!(ValDataLoader)
        PythonCall.pydel!(TestDataLoader)

        open(joinpath(folder, "optuna.txt"), "a") do f
            write(f, "0.0")
        end
        exit(0)
    end

    if WORKERS
        try
            TrainDataLoader._iterator._shutdown_workers()
        catch e
            @show e
            @info "Could not shutdown train workers"
        end
        try
            ValDataLoader._iterator._shutdown_workers()
        catch e
            @show e
            @info "Could not shutdown val workers"
        end
    end

    save_distill_weights!(outs, folder)
    PythonCall.pydel!(TrainDataset)
    PythonCall.pydel!(ValDataset)
    PythonCall.pydel!(TestDataset)
    PythonCall.pydel!(TrainDataLoader)
    PythonCall.pydel!(ValDataLoader)
    PythonCall.pydel!(TestDataLoader)
    dataloader_setup = nothing
    pylipext.run_gc(_pylipbackend)
    exit(0)
end

load_distill_weights!(pybackend, model_tail, model_head, folder)
image_arrays = build_distill_image_arrays(trainx, valx, testx)
vec_of_3D_imgs_train = image_arrays.vec_of_3d_imgs_train
vec_of_3D_imgs_val = image_arrays.vec_of_3d_imgs_val
vec_of_3D_imgs_test = image_arrays.vec_of_3d_imgs_test

eval_setup = build_eval_distill_dataloaders(
    pylipext,
    vec_of_3D_imgs_train,
    trainy,
    vec_of_3D_imgs_val,
    valy,
    vec_of_3D_imgs_test,
    testy,
    py_val_transforms,
    BS_TRAIN,
    BS_VAL
)
TrainDataset = eval_setup.train_dataset
ValDataset = eval_setup.val_dataset
TestDataset = eval_setup.test_dataset
TrainDataLoader = eval_setup.train_dataloader
ValDataLoader = eval_setup.val_dataloader
TestDataLoader = eval_setup.test_dataloader

SUBTRACT_VAL_ARGMAX = L1_OUT == 2 ? 0.0 : 1.0 # (0,1) => (0, 1) or (0,1,2) => (-1,0,1)
if BINARIZE
    final_model = pylipext._get_gumbelchainreshape_layer(_pylipbackend, model_tail.nn, model_head.nn, L1, L1_OUT, SUBTRACT_VAL_ARGMAX)
else
    final_model = pylipext._get_chain_layer(_pylipbackend, model_tail.nn, model_head.nn)
end
n_classes = N_CLASSES
train_bacc, _ = MAGENetwork.test_parallel_model(
    _pylipbackend,
    final_model,
    TrainDataLoader
)
@info train_bacc
val_bacc, _ = MAGENetwork.test_parallel_model(
    _pylipbackend,
    final_model,
    ValDataLoader
)
@info val_bacc

test_bacc, _ = MAGENetwork.test_parallel_model(
    _pylipbackend,
    final_model,
    TestDataLoader
)
@info test_bacc

if IS_PARAM_TUNING
    PythonCall.pydel!(TrainDataLoader)
    PythonCall.pydel!(ValDataLoader)
    PythonCall.pydel!(TestDataLoader)
    exit(0)
end

# CAPTURE TRAIN
captured_train_data, idx_train = pylipext.capture_surrogate_training_data(
    _pylipbackend, final_model, TrainDataLoader; binarize = BINARIZE
)
captured_val_data, idx_val = pylipext.capture_surrogate_training_data(
    _pylipbackend, final_model, ValDataLoader; binarize = BINARIZE
)

# Don't SAVE models that are constant
isconstant = Dict()
for (k, v) in captured_train_data
    isconstant[k] = false
    if std(map(x -> x.outputs, v)) ≈ 0.0
        isconstant[k] = true
    end
end
for (k, v) in captured_val_data
    if std(map(x -> x.outputs, v)) ≈ 0.0
        isconstant[k] = true
    end
end
@show isconstant # if true it was constant in train and/or val

"""Compute a numerically safe Pearson correlation, returning `NaN` for degenerate vectors."""
function safe_pearson_cor(xs, ys)
    x = Float64.(xs)
    y = Float64.(ys)
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    denom = sqrt(sum(abs2, x_centered) * sum(abs2, y_centered))
    return denom == 0.0 ? NaN : sum(x_centered .* y_centered) / denom
end

"""Print per-class correlation and mean-activation summaries for one latent neuron."""
function print_latent_class_correlations(ys, labels, classes, split_str, neuron_idx)
    y_values = Float64.(ys)
    class_correlations = Pair{Any, Float64}[]
    class_means = Pair{Any, Float64}[]
    for cls in classes
        mask = labels .== cls
        push!(class_correlations, cls => safe_pearson_cor(y_values, Float64.(mask)))
        push!(class_means, cls => any(mask) ? mean(y_values[mask]) : NaN)
    end
    sorted_correlations = sort(class_correlations; by = p -> (isnan(p.second) ? -Inf : abs(p.second)), rev = true)
    sorted_means = sort(class_means; by = p -> (isnan(p.second) ? -Inf : p.second), rev = true)
    println("Latent neuron class correlations | split=$split_str | neuron_idx=$neuron_idx")
    println("class_correlations = $(repr(class_correlations))")
    println("sorted_by_abs_correlation = $(repr(sorted_correlations))")
    println("Latent neuron class activation means | split=$split_str | neuron_idx=$neuron_idx")
    println("class_activation_means = $(repr(class_means))")
    return println("sorted_by_mean_activation = $(repr(sorted_means))")
end

# SAVE ONE BY ONE
for (split, real_x, real_y, split_str) in ((captured_train_data, trainx, trainy, "train"), (captured_val_data, valx, valy, "val"))
    xs = map(v -> map(i -> reinterpret.(UInt8, i.img), v), real_x)
    for i in 1:L1 # only save first layer
        k = (1, i)
        ys = map(x -> x.outputs, split[k])
        println(histogram(ys))
        print_latent_class_correlations(ys, real_y, CLASSES, split_str, i)
        @assert length(ys) == length(xs)
        tmp = (
            means = means,
            stds = stds,
            xs = xs,
            ys = ys,
            extras = (
                :struct_ => SurrogateDataset_IMG_SCALAR,
            ),
        )
        if isconstant[k] && split_str == "train"
            @warn "Module $k is constant"
            open(joinpath(folder, "ignore_modules.txt"), "a") do f
                write(f, "$(k[1]),$(k[2])\n")
            end
        end
        p = joinpath(folder, "surrogate_$(k[1])_$(k[2])_$(split_str).jld2")
        save_object(p, tmp)
        @info "Data saved for $k for $split_str : $(length(xs))"
    end
end

open(joinpath(folder, "metrics.txt"), "w") do f
    write(f, "$train_bacc, $val_bacc, $test_bacc\n")
end

# LOAD WEIGHTS WITH
# @pyexec (
#     m = tail.nn,
#     torch = pylipbackend.torch, filename = joinpath(folder, "model_weights_tail(1).pt"), loc = "cpu",
# ) => "m.load_state_dict(torch.load(filename, map_location = loc))"

# @pyeval (
#     m = head.nn,
#     torch = pylipbackend.torch, filename = joinpath(folder, "model_weights_head(2).pt"), loc = "cpu",
# ) => "m.load_state_dict(torch.load(filename, map_location = loc))"

# FROM CHANNELS TO COLOR VIEW

# function to_rgb(x) # for vector of simageND
#     xs = [reinterpret(z.img) for z in x]
#     return colorview(RGB, permutedims(cat(xs..., dims = 3), (3, 1, 2)))
# end

# function to_rgb(xs...) # for vector of uint8
#     xs = [reinterpret.(N0f8, x) for x in xs]
#     return colorview(RGB, permutedims(cat(xs..., dims = 3), (3, 1, 2)))
# end


# detect memory issues
# for (batch_idx, pack) in enumerate(TrainDataLoader)
#     idx, inputs, labels = pack
#     inputs_py, labels_py = inputs.to("cuda:0"), labels.to("cuda:0")
#     inputs_py * 2
#     @show inputs_py[1]
#     PythonCall.pydel!(pack)
# end
