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
    isfile(tail_file) || error("Missing tail weights file: $tail_file")
    isfile(head_file) || error("Missing head weights file: $head_file")
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

"""Best-effort shutdown of dataloader worker processes to release Python-side resources."""
function shutdown_dataloader_workers!(dl, name::String)
    try
        dl._iterator._shutdown_workers()
    catch e
        @show e
        @info "Could not shutdown $name workers"
    end
    return
end

"""Remove `--distill_phase` options from raw CLI args before forwarding to subprocesses."""
function strip_distill_phase_args(args::Vector{String})
    arg_string = join(copy(args), " ")
    arg_string = replace(arg_string, r"--distill_phase=\S+" => "")
    arg_string = replace(arg_string, r"--distill_phase\s+\S+" => "")
    return isempty(strip(arg_string)) ? String[] : split(strip(arg_string))
end
