using CUDA, cuDNN
using PartialFunctions
using PythonCall
import MLJ
using UnicodePlots
using Statistics
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
include(joinpath(home, "src", "distill", "common.jl"))
include(joinpath(home, "src", "distill", "data.jl"))
include(joinpath(home, "src", "distill", "mage_stuff.jl"))

parsed_args = parse_distill_args(default_phase = "train_only")
parsed_args["distill_phase"] = "train_only"
@show parsed_args

mix_prob = parsed_args["mix_prob"]
lambda_student = parsed_args["lambda_student"]
min_student_loss = parsed_args["min_student_loss"]
min_entropy_loss = parsed_args["min_entropy_loss"]
weight_entropy = parsed_args["weight_entropy"]
dropout = parsed_args["dropout"]
weight_decay = parsed_args["wd"]
binarize = parsed_args["binarize"]
optim = parsed_args["optim"]
last_layer_type = Symbol(parsed_args["last_layer_type"])
workers = parsed_args["workers"]
pretrained_model = parsed_args["pretrained"]
tail_activation = lowercase(parsed_args["tail_activation"])
torchvision_weights = parsed_args["torchvision_weights"]
use_imagenet_stats = parsed_args["use_imagenet_stats"]
resize_to = parsed_args["resize_to"]
lip = parsed_args["lip"]
schedule = parsed_args["schedule"]
boost_round = parsed_args["boost_round"]
epochs = parsed_args["epochs"]
train_bs = parsed_args["train_bs"]
val_bs = parsed_args["val_bs"]
student_arch = Symbol(parsed_args["student_arch"])
student_inner_channels = parsed_args["student_inner_channels"]
student_n_mlp = parsed_args["student_n_mlp"]
student_dropout = parsed_args["student_dropout"]

@info "BOOSTING ROUND $boost_round"
include(joinpath(home, "src", "magenet_ski.jl"))

const_vars = setup_constants(parsed_args)
map(i -> eval(i), const_vars)
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images
include_utils_and_disable_logging(home)

parsed_args["type_of_module"] = "v1"
type_of_module = Symbol(parsed_args["type_of_module"])
MAGENetwork.DP_RATE[] = dropout
MAGENetwork.OPTIM[] = OPTIM
MAGENetwork.MODULE_ACT[] = resolve_tail_activation(tail_activation)
MAGENetwork.INTER_MODULE_ACT[] = relu
@info "Tail activation" tail_activation = tail_activation module_act = MAGENetwork.MODULE_ACT[]

trainx, trainy, valx, valy, testx, testy = load_distill_split_arrays(parsed_args)
classes, n_classes, sample_img, n_ins = infer_distill_dataset_metadata(trainx, trainy)
means, stds = compute_distill_channel_stats(trainx)
transform_setup = build_distill_transforms(
    sample_img,
    n_ins;
    pretrained_model = pretrained_model,
    use_imagenet_stats = use_imagenet_stats,
    resize_to = resize_to,
    means = means,
    stds = stds,
)
means = transform_setup.means
stds = transform_setup.stds
img_size = transform_setup.img_size
py_train_transforms = transform_setup.py_train_transforms
py_val_transforms = transform_setup.py_val_transforms
@info "BS : $train_bs $val_bs"

setup_distill_image_bundles!(home, sample_img)
float_bundles, only_float_bundles = create_distill_float_bundles()
ml, ml_float = create_distill_metalibraries(float_bundles, only_float_bundles)
pybackend, pylipbackend, pylipext = create_distill_py_backends(lip)
log_distill_torch_status(pybackend)
seed_distill_torch!(pybackend, SEED)
initial_pop = create_distill_initial_population(pybackend, parsed_args, ml, ml_float, valx, n_classes, type_of_module)

model_tail, model_head, l1, l1_out = create_distill_models(
    pylipext,
    pybackend,
    parsed_args;
    n_ins = n_ins,
    n_classes = n_classes,
    binarize = binarize,
    tail_activation_name = tail_activation,
    torchvision_weights_arg = torchvision_weights,
    last_layer_type = last_layer_type,
)

folder = ensure_distill_output_folder(parsed_args["output_dir"], parsed_args["trial_id"], boost_round)

dataloader_setup = build_distill_dataloaders(
    pylipext,
    trainx, trainy,
    valx, valy,
    testx, testy,
    py_train_transforms,
    py_val_transforms,
    train_bs,
    val_bs;
    workers = workers,
)
train_dataset = dataloader_setup.train_dataset
val_dataset = dataloader_setup.val_dataset
test_dataset = dataloader_setup.test_dataset
train_dataloader = dataloader_setup.train_dataloader
val_dataloader = dataloader_setup.val_dataloader
test_dataloader = dataloader_setup.test_dataloader

# Warm up Python datasets to avoid first-access issues in worker processes.
@pyeval (x = train_dataset,) => "x[1]"
@pyeval (x = val_dataset,) => "x[1]"
@pyeval (x = test_dataset,) => "x[1]"

pybackend.torch.backends.cudnn.benchmark = false
torchsummary = pyimport("torchsummary")
torchsummary.summary(model_tail.nn, (n_ins, img_size, img_size), depth = 5);
torchsummary.summary(model_head.nn, (l1,), depth = 3);

best_lr = parsed_args["lr"]
@show best_lr
MAGENetwork.OPTIM[] = optim
MAGENetwork.LR[] = best_lr
pybackend.torch.set_float32_matmul_precision("high")
use_students = pylipext.WithoutStudent()

outs = try
    MAGENetwork.train_tail_and_heads(
        pylipbackend, initial_pop[1],
        model_tail, model_head;
        binarize = binarize,
        n_classes = n_classes,
        max_epochs = epochs,
        train_dl = train_dataloader,
        val_dl = val_dataloader,
        to_schedule = schedule,
        to_penalise_sim = false,
        img_size = img_size,
        mix_prob = mix_prob,
        activation_fn = nothing,
        lambda_cls = 1.0,
        label_smoothing_factor = 0.0,
        alpha_decode = 0.0,
        weight_decay = weight_decay,
        neurons_class = l1,
        neurons_per_class = l1_out,
        n_channels = n_ins,
        use_students = use_students,
        student_arch = student_arch,
        student_inner_channels_cnn = student_inner_channels,
        student_n_mlp = student_n_mlp,
        student_dropout = student_dropout,
        student_agg = :sum,
        lambda_student = lambda_student,
        min_student_loss = min_student_loss,
        use_student_entropy = true,
        min_entropy_loss = min_entropy_loss,
        weight_entropy = weight_entropy,
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
    workers && shutdown_dataloader_workers!(train_dataloader, "train")
    workers && shutdown_dataloader_workers!(val_dataloader, "val")
    PythonCall.pydel!(train_dataset)
    PythonCall.pydel!(val_dataset)
    PythonCall.pydel!(test_dataset)
    PythonCall.pydel!(train_dataloader)
    PythonCall.pydel!(val_dataloader)
    PythonCall.pydel!(test_dataloader)
    open(joinpath(folder, "optuna.txt"), "a") do f
        write(f, "0.0")
    end
    exit(1)
end

workers && shutdown_dataloader_workers!(train_dataloader, "train")
workers && shutdown_dataloader_workers!(val_dataloader, "val")
save_distill_weights!(outs, folder)
PythonCall.pydel!(train_dataset)
PythonCall.pydel!(val_dataset)
PythonCall.pydel!(test_dataset)
PythonCall.pydel!(train_dataloader)
PythonCall.pydel!(val_dataloader)
PythonCall.pydel!(test_dataloader)
dataloader_setup = nothing
pylipext.run_gc(pylipbackend)
