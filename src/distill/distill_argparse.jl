"""Parse distillation CLI arguments and apply shared distill argument normalization."""
function parse_distill_args(; default_phase::String = "all")
    local_s = deepcopy(s)
    @add_arg_table local_s begin
        "--boost_round"
        arg_type = Int
        "--latent_dim"
        arg_type = Int
        default = 5
        "--lip"
        arg_type = Bool
        default = false
        "--param_tuning"
        arg_type = Bool
        default = false
        "--val_bs"
        arg_type = Int
        default = 512
        "--workers"
        arg_type = Bool
        default = false
        "--pretrained"
        arg_type = String
        default = "resnet18"
        "--tail_activation"
        arg_type = String
        default = "identity"
        "--torchvision_weights"
        arg_type = String
        default = "DEFAULT"
        "--distill_phase"
        arg_type = String
        default = default_phase
        "--use_imagenet_stats"
        arg_type = Bool
        default = false
        "--resize_to"
        arg_type = Int
        default = -1
        "--train_bs"
        arg_type = Int
        default = 128
        "--dropout"
        arg_type = Float64
        default = 0.2
        "--last_layer_type"
        arg_type = String
        default = "vsmall"
        "--lr"
        arg_type = Float64
        default = 0.001
        "--wd"
        arg_type = Float64
        default = 0.01
        "--schedule"
        arg_type = Bool
        default = true
        "--mix_prob"
        arg_type = Float64
        default = 0.5
        "--binarize"
        arg_type = Bool
        default = false
        "--min_entropy_loss"
        arg_type = Float64
        default = 1.0
        "--weight_entropy"
        arg_type = Float64
        default = 1.0
        "--num_student_type"
        arg_type = String
        default = "without_student"
        "--student_arch"
        arg_type = String
        default = "cnn"
        "--student_inner_channels"
        arg_type = Int
        default = 1
        "--student_n_mlp"
        arg_type = Int
        default = 1
        "--student_dropout"
        arg_type = Float64
        default = 0.2
        "--lambda_student"
        arg_type = Float64
        default = 0.5
        "--min_student_loss"
        arg_type = Float64
        default = 0.05
    end
    parsed_args = parse_args(local_s)
    normalize_distill_args!(parsed_args)
    return parsed_args
end
