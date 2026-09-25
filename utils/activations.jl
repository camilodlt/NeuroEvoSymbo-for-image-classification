import Flux

abstract type AbstractInterActivation end

struct TanhActivation <: AbstractInterActivation end
struct AsinhActivation <: AbstractInterActivation end
struct SigmoidActivation <: AbstractInterActivation end
struct IdentityActivation <: AbstractInterActivation end
struct ArgmaxActivation <: AbstractInterActivation end
struct PositiveClassActivation <: AbstractInterActivation end
struct PositiveClassRelativeActivation <: AbstractInterActivation end
struct SoftmaxPositiveClassActivation <: AbstractInterActivation end

function _require_binary_outputs(act::AbstractInterActivation, x::AbstractArray{T}) where {T <: Number}
    @assert length(x) == 2 "$(activation_name(act)) requires exactly 2 outputs, got $(length(x))"
    return x
end

activation_name(::TanhActivation) = "tanh"
activation_name(::AsinhActivation) = "asinh"
activation_name(::SigmoidActivation) = "sigmoid"
activation_name(::IdentityActivation) = "identity"
activation_name(::ArgmaxActivation) = "argmax"
activation_name(::PositiveClassActivation) = "positive_class"
activation_name(::PositiveClassRelativeActivation) = "positive_class_relative"
activation_name(::SoftmaxPositiveClassActivation) = "softmax_positive_class"

requires_binary_outputs(::AbstractInterActivation) = false
requires_binary_outputs(::PositiveClassActivation) = true
requires_binary_outputs(::PositiveClassRelativeActivation) = true
requires_binary_outputs(::SoftmaxPositiveClassActivation) = true

supports_scalar_output(::AbstractInterActivation) = false
supports_scalar_output(::TanhActivation) = true
supports_scalar_output(::AsinhActivation) = true
supports_scalar_output(::SigmoidActivation) = true
supports_scalar_output(::IdentityActivation) = true

(::TanhActivation)(nb::Float64) = Flux.tanh(nb)
(::TanhActivation)(x::AbstractArray{T}) where {T <: Number} = Flux.tanh.(x)

(::AsinhActivation)(nb::Float64) = asinh(nb)
(::AsinhActivation)(x::AbstractArray{T}) where {T <: Number} = asinh.(x)

(::SigmoidActivation)(nb::Float64) = Flux.sigmoid(nb)
(::SigmoidActivation)(x::AbstractArray{T}) where {T <: Number} = Flux.sigmoid.(x)

(::IdentityActivation)(nb::Float64) = identity(nb)
(::IdentityActivation)(x::AbstractArray{T}) where {T <: Number} = identity.(x)

(::ArgmaxActivation)(x::AbstractArray{T}) where {T <: Number} = argmax(x) - 1
function (::ArgmaxActivation)(x::NTuple{N, T}) where {N, T <: Number}
    @assert N >= 1 "argmax requires at least one output, got $N"
    return argmax(x) - 1
end

function (act::PositiveClassActivation)(x::AbstractArray{T}) where {T <: Number}
    _require_binary_outputs(act, x)
    return x[end]
end
function (act::PositiveClassActivation)(x::NTuple{2, T}) where {T <: Number}
    return x[end]
end

function (act::PositiveClassRelativeActivation)(x::AbstractArray{T}) where {T <: Number}
    _require_binary_outputs(act, x)
    return x[end] / (x[begin] + eps(Float64))
end
function (act::PositiveClassRelativeActivation)(x::NTuple{2, T}) where {T <: Number}
    return x[end] / (x[begin] + eps(Float64))
end

function (act::SoftmaxPositiveClassActivation)(x::AbstractArray{T}) where {T <: Number}
    _require_binary_outputs(act, x)
    return Flux.softmax(x)[end]
end
function (act::SoftmaxPositiveClassActivation)(x::NTuple{2, T}) where {T <: Number}
    return Flux.softmax(collect(x))[end]
end

function activation_from_name(act_name::String)
    if act_name == "tanh"
        return TanhActivation()
    elseif act_name == "asinh"
        return AsinhActivation()
    elseif act_name == "sigmoid"
        return SigmoidActivation()
    elseif act_name == "identity"
        return IdentityActivation()
    elseif act_name == "argmax"
        return ArgmaxActivation()
    elseif act_name == "positive_class"
        return PositiveClassActivation()
    elseif act_name == "positive_class_relative"
        return PositiveClassRelativeActivation()
    elseif act_name == "softmax_positive_class"
        return SoftmaxPositiveClassActivation()
    else
        error("NO ACTIVATION $act_name")
    end
end

build_inter_act(act_name::String) = activation_from_name(act_name)

function supported_all_activations()
    return AbstractInterActivation[
        TanhActivation(),
        AsinhActivation(),
        SigmoidActivation(),
        IdentityActivation(),
        ArgmaxActivation(),
        PositiveClassActivation(),
        PositiveClassRelativeActivation(),
        SoftmaxPositiveClassActivation(),
    ]
end

function supported_regression_activations()
    return AbstractInterActivation[
        TanhActivation(),
        AsinhActivation(),
        SigmoidActivation(),
        IdentityActivation(),
    ]
end

function supported_classification_activations()
    return AbstractInterActivation[
        ArgmaxActivation(),
        PositiveClassActivation(),
        PositiveClassRelativeActivation(),
        SoftmaxPositiveClassActivation(),
    ]
end

function supported_conv_classification_activations()
    return supported_classification_activations()
end

function activation_is_applicable(act::AbstractInterActivation, n_outputs::Integer)
    if n_outputs == 1
        return supports_scalar_output(act)
    end
    if requires_binary_outputs(act)
        return n_outputs == 2
    end
    return n_outputs >= 2
end

function candidate_activations_for_family(act_family::String)
    if act_family == "all"
        return supported_all_activations()
    elseif act_family == "regression"
        return supported_regression_activations()
    elseif act_family == "classification"
        return supported_classification_activations()
    else
        error("Unsupported act family $act_family. Expected one of: all, regression, classification")
    end
end

function compatible_activations_for_output_length(n_outputs::Integer; act_family::String = "all")
    candidates = candidate_activations_for_family(act_family)
    return filter(act -> activation_is_applicable(act, n_outputs), candidates)
end

act_to_use = haskey(Parsed_args, "act") ? Parsed_args["act"] : "identity"
@info "USING ACT : $act_to_use"
INTER_ACT = Ref{AbstractInterActivation}()
INTER_ACT[] = build_inter_act(act_to_use)
