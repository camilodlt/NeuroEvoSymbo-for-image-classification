"""Initialize image function bundles used by MAGE programs for distillation."""
function setup_distill_image_bundles!(home, sample_img)
    include(joinpath(home, "src", "magenet_image_bundles.jl"))
    define_common_image_functions(sample_img)
    return
end

"""Create float bundles and set deterministic casters for both full and SR libraries."""
function create_distill_float_bundles()
    float_bundles = UTCGP.get_float_bundles()
    only_float_bundles = UTCGP.get_sr_float_bundles()
    set_bundle_casters!(float_bundles, float_caster2)
    set_bundle_casters!(only_float_bundles, float_caster2)
    return float_bundles, only_float_bundles
end

"""Build the main and float-only metalibraries used during distillation."""
function create_distill_metalibraries(float_bundles, only_float_bundles)
    ml = ml_from_vbundles([image_intensity, image_binary, image_segment, float_bundles])
    ml_float = ml_from_vbundles([only_float_bundles])
    return ml, ml_float
end

"""Create PyTorch backends and resolve the PytorchLip extension handle."""
function create_distill_py_backends(lip::Bool)
    pylipbackend = MAGENetwork.get_pytorchlip_backend()
    pybackend = lip ? pylipbackend : MAGENetwork.get_pytorch_backend()
    pylipext = Base.get_extension(MAGENetwork, :PytorchLip)
    return pybackend, pylipbackend, pylipext
end

"""Log Torch CUDA availability details to verify runtime device selection."""
function log_distill_torch_status(pybackend)
    torch_cuda_available = pyconvert(Bool, pybackend.torch.cuda.is_available())
    torch_cuda_device_count = pyconvert(Int, pybackend.torch.cuda.device_count())
    torch_cuda_current_device = torch_cuda_available ? pyconvert(Int, pybackend.torch.cuda.current_device()) : -1
    @info "Torch CUDA status" available = torch_cuda_available device_count = torch_cuda_device_count current_device = torch_cuda_current_device
    return
end

"""Seed Torch RNG for reproducible model initialization and training behavior."""
function seed_distill_torch!(pybackend, seed::Int)
    @info "Setting TORCH seed to $seed"
    pybackend.torch.manual_seed(seed)
    return
end

"""Build the initial MAGE population used to define the distillation architecture."""
function create_distill_initial_population(pybackend, parsed_args, ml, ml_float, valx, n_classes::Int, type_of_module::Symbol)
    return create_initial_magenet_population(
        pybackend,
        1,
        parsed_args,
        Type2Dimg_intensity,
        (Type2Dimg_binary, Type2Dimg_segment),
        ml,
        ml_float,
        valx,
        20,
        n_classes;
        type_of_module,
        use_surrogate = false
    )
end

"""Compute latent layout dimensions used to reshape tail outputs before the head."""
function compute_distill_l1_dims(parsed_args, binarize::Bool)
    l1 = parsed_args["l1_size"]
    l1_out = parsed_args["l1_out_size"]
    l1_out = binarize ? l1_out : 1
    return l1, l1_out
end

"""Create argument metadata needed to instantiate the image-to-latent tail model."""
function create_distill_tail_args(pybackend, n_ins::Int, nneurons_total::Int)
    ma_tail = modelArchitecture(
        [[Type2Dimg_intensity for _ in 1:n_ins]...],
        [[1 for _ in 1:n_ins]...],
        [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment, Float64],
        [Float64 for _ in 1:nneurons_total],
        [4 for _ in 1:nneurons_total]
    )
    return MAGENetwork._args_needed_per_model(MAGENetwork.ImagesToScalarNN, ma_tail, pybackend)
end

"""Create argument metadata needed to instantiate the latent-to-class head model."""
function create_distill_head_args(pybackend, l1::Int, n_classes::Int)
    ma_head = modelArchitecture(
        [Float64 for _ in 1:l1],
        [1 for _ in 1:l1],
        [Float64],
        [Float64 for _ in 1:n_classes],
        [1 for _ in 1:n_classes]
    )
    return MAGENetwork._args_needed_per_model(MAGENetwork.ScalarsToScalarNN, ma_head, pybackend)
end

"""Instantiate the tail model, selecting pretrained backbone or default custom architecture."""
function create_distill_tail_model(pylipext, pybackend, model_args_tail, parsed_args, n_ins::Int, nneurons_total::Int, tail_activation_name::String, torchvision_weights_arg::String)
    pretrained_model = parsed_args["pretrained"]
    tail_activation = resolve_tail_activation(pybackend, tail_activation_name)
    torchvision_weights = resolve_torchvision_weights(torchvision_weights_arg)
    @info "Torchvision weights" torchvision_weights

    if pretrained_model == "resnext50_32x4d"
        _model = pylipext.get_resnext50_32x4d()(
            new_hidden_dim = nneurons_total,
            weights = torchvision_weights,
            freeze_backbone = false,
            dropout_mlp = MAGENetwork.DP_RATE[],
            mlp_hidden_layers = [256, 128],
            n_channels_in = n_ins,
            activation = tail_activation
        )
        return MAGENetwork.ImagesToScalarNN(_model, model_args_tail..., :notdefined, false, pybackend)
    elseif occursin("resnet", pretrained_model)
        _model = pylipext.get_resnet()(
            new_hidden_dim = nneurons_total,
            backbone_name = pretrained_model,
            weights = torchvision_weights,
            freeze_backbone = false,
            dropout_mlp = MAGENetwork.DP_RATE[],
            mlp_hidden_layers = [256, 128],
            n_channels_in = n_ins,
            activation = tail_activation
        )
        return MAGENetwork.ImagesToScalarNN(_model, model_args_tail..., :notdefined, false, pybackend)
    elseif pretrained_model == "vit"
        _model = pylipext.get_vit()(
            new_hidden_dim = nneurons_total,
            backbone_name = "vit_b_16",
            weights = "DEFAULT",
            freeze_backbone = false,
            dropout_mlp = MAGENetwork.DP_RATE[],
            mlp_hidden_layers = [192],
            n_channels_in = n_ins,
            activation = tail_activation
        )
        return MAGENetwork.ImagesToScalarNN(_model, model_args_tail..., :notdefined, false, pybackend)
    else
        return MAGENetwork.create_nn_model(pybackend, MAGENetwork.ImagesToScalarNN, model_args_tail...; size = :gumbel_softmax, last = false)
    end
end

"""Instantiate the classification head model using the configured last-layer family."""
function create_distill_head_model(pybackend, model_args_head, last_layer_type::Symbol)
    return MAGENetwork.create_nn_model(pybackend, MAGENetwork.ScalarsToScalarNN, model_args_head...; size = last_layer_type, last = true)
end

"""Build tail/head models and latent dimensions for distillation from parsed settings."""
function create_distill_models(pylipext, pybackend, parsed_args; n_ins::Int, n_classes::Int, binarize::Bool, tail_activation_name::String, torchvision_weights_arg::String, last_layer_type::Symbol)
    l1, l1_out = compute_distill_l1_dims(parsed_args, binarize)
    nneurons_total = l1 * l1_out
    @info "NN neurons : $(nneurons_total) : L1 $l1 and L1_OUT $l1_out. It will be reshaped to ($l1, $l1_out)"
    model_args_tail = create_distill_tail_args(pybackend, n_ins, nneurons_total)
    model_args_head = create_distill_head_args(pybackend, l1, n_classes)
    @info "Model size type : :gumbel_softmax"
    model_tail = create_distill_tail_model(pylipext, pybackend, model_args_tail, parsed_args, n_ins, nneurons_total, tail_activation_name, torchvision_weights_arg)
    model_head = create_distill_head_model(pybackend, model_args_head, last_layer_type)
    return model_tail, model_head, l1, l1_out
end
