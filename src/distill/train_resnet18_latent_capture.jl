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

parsed_args = parse_distill_args(default_phase = "capture_only")
parsed_args["distill_phase"] = "capture_only"
@show parsed_args

binarize = parsed_args["binarize"]
last_layer_type = Symbol(parsed_args["last_layer_type"])
pretrained_model = parsed_args["pretrained"]
tail_activation = lowercase(parsed_args["tail_activation"])
torchvision_weights = parsed_args["torchvision_weights"]
use_imagenet_stats = parsed_args["use_imagenet_stats"]
resize_to = parsed_args["resize_to"]
lip = parsed_args["lip"]
boost_round = parsed_args["boost_round"]
is_param_tuning = parsed_args["param_tuning"]
train_bs = parsed_args["train_bs"]
val_bs = parsed_args["val_bs"]

@info "BOOSTING ROUND $boost_round"
include(joinpath(home, "src", "magenet_ski.jl"))

const_vars = setup_constants(parsed_args)
map(i -> eval(i), const_vars)
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images
include_utils_and_disable_logging(home)

parsed_args["type_of_module"] = "v1"
type_of_module = Symbol(parsed_args["type_of_module"])
MAGENetwork.DP_RATE[] = parsed_args["dropout"]
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
py_val_transforms = transform_setup.py_val_transforms
@info "BS : $train_bs $val_bs"

setup_distill_image_bundles!(home, sample_img)
float_bundles, only_float_bundles = create_distill_float_bundles()
ml, ml_float = create_distill_metalibraries(float_bundles, only_float_bundles)
pybackend, pylipbackend, pylipext = create_distill_py_backends(lip)
log_distill_torch_status(pybackend)
seed_distill_torch!(pybackend, SEED)
initial_pop = create_distill_initial_population(pybackend, parsed_args, ml, ml_float, valx, n_classes, type_of_module)
_ = initial_pop

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
load_distill_weights!(pybackend, model_tail, model_head, folder)

image_arrays = build_distill_image_arrays(trainx, valx, testx)
eval_setup = build_eval_distill_dataloaders(
    pylipext,
    image_arrays.vec_of_3d_imgs_train,
    trainy,
    image_arrays.vec_of_3d_imgs_val,
    valy,
    image_arrays.vec_of_3d_imgs_test,
    testy,
    py_val_transforms,
    train_bs,
    val_bs
)
train_dataset = eval_setup.train_dataset
val_dataset = eval_setup.val_dataset
test_dataset = eval_setup.test_dataset
train_dataloader = eval_setup.train_dataloader
val_dataloader = eval_setup.val_dataloader
test_dataloader = eval_setup.test_dataloader

# Warm up Python datasets to avoid first-access issues in worker processes.
@pyeval (x = train_dataset,) => "x[1]"
@pyeval (x = val_dataset,) => "x[1]"
@pyeval (x = test_dataset,) => "x[1]"

subtract_val_argmax = l1_out == 2 ? 0.0 : 1.0
final_model = if binarize
    pylipext._get_gumbelchainreshape_layer(
        pylipbackend,
        model_tail.nn,
        model_head.nn,
        l1,
        l1_out,
        subtract_val_argmax
    )
else
    pylipext._get_chain_layer(pylipbackend, model_tail.nn, model_head.nn)
end

train_bacc, _ = MAGENetwork.test_parallel_model(pylipbackend, final_model, train_dataloader)
@info train_bacc
val_bacc, _ = MAGENetwork.test_parallel_model(pylipbackend, final_model, val_dataloader)
@info val_bacc
test_bacc, _ = MAGENetwork.test_parallel_model(pylipbackend, final_model, test_dataloader)
@info test_bacc

if !is_param_tuning
    captured_train_data, _ = pylipext.capture_surrogate_training_data(
        pylipbackend,
        final_model,
        train_dataloader;
        binarize = binarize
    )
    captured_val_data, _ = pylipext.capture_surrogate_training_data(
        pylipbackend,
        final_model,
        val_dataloader;
        binarize = binarize
    )

    isconstant = Dict()
    for (k, v) in captured_train_data
        isconstant[k] = std(map(x -> x.outputs, v)) ≈ 0.0
    end
    for (k, v) in captured_val_data
        isconstant[k] = get(isconstant, k, false) || (std(map(x -> x.outputs, v)) ≈ 0.0)
    end
    @show isconstant

    for (split, real_x, real_y, split_str) in (
        (captured_train_data, trainx, trainy, "train"),
        (captured_val_data, valx, valy, "val"),
    )
        xs = map(v -> map(i -> reinterpret.(UInt8, i.img), v), real_x)
        for i in 1:l1
            k = (1, i)
            ys = map(x -> x.outputs, split[k])
            println(histogram(ys))
            print_latent_class_correlations(ys, real_y, classes, split_str, i)
            @assert length(ys) == length(xs)
            tmp = (
                means = means,
                stds = stds,
                xs = xs,
                ys = ys,
                extras = (:struct_ => SurrogateDataset_IMG_SCALAR,),
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
end

open(joinpath(folder, "metrics.txt"), "w") do f
    write(f, "$train_bacc, $val_bacc, $test_bacc\n")
end

PythonCall.pydel!(train_dataset)
PythonCall.pydel!(val_dataset)
PythonCall.pydel!(test_dataset)
PythonCall.pydel!(train_dataloader)
PythonCall.pydel!(val_dataloader)
PythonCall.pydel!(test_dataloader)
pylipext.run_gc(pylipbackend)
