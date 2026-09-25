"""
Per-class sensitivity and specificity table.

Collects per-class sensitivity (true positive rate) and specificity (true negative rate)
on the test split for every method of the main comparison and writes a tidy CSV and a
LaTeX table. All rates are computed with `StatisticalMeasures.MulticlassTruePositiveRate`
and `StatisticalMeasures.MulticlassTrueNegativeRate` under `average = NoAvg()`.

Columns, in order:
  - end-to-end MAGE (validation-selected run)
  - distilled MAGE: three dictionaries (best / runs / all per dimension) x three heads (RF, SVM, LR)
  - neural baselines: ResNet18, ResNet34, ResNeXt50

Sources:
  - distilled heads : <output_dir>/<trial_id>/ml_models_boost_<r>/extended_metrics_<metric>_<suffix>.csv,
                      whose `sensitivity` and `specificity` columns already hold the per-class rates
  - end-to-end MAGE : <output_dir>/<e2e_trial>/mage_imgcls/per_class_test_all{false,true}.csv,
                      written by src/read_best_magecls_per_trial.jl
  - neural baselines: <output_dir>/NN_baselines/<backbone>_<dataset>/per_class_test.csv,
                      written here with --nn true, which loads the best checkpoint through
                      python_utils/medmnist_resnet_baseline.py and scores the test split

Missing sources are reported as `--`, so the table can be produced incrementally.

Example:
  julia --project plots/per_class_metrics.jl \
    --output_dir BloodMNIST --trial_id L16_WD01_asinh_R18_128 \
    --dataset BloodMNIST --num_classes 8 --nd 3 --metric macro_f1 --nn true \
    --class_names "basophil,eosinophil,erythroblast,immature gran.,lymphocyte,monocyte,neutrophil,platelet"
"""

using ArgParse
using CSV
using DataFrames
using Printf
using PythonCall
using StatisticalMeasures

function build_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--output_dir"
        arg_type = String
        required = true
        "--trial_id"
        arg_type = String
        required = true
        "--dataset"
        help = "MedMNIST name used in the NN_baselines folder names, e.g. BloodMNIST"
        arg_type = String
        required = true
        "--num_classes"
        arg_type = Int
        required = true
        "--nd"
        help = "1 for grayscale, 3 for colour; only used with --nn true"
        arg_type = Int
        default = 3
        "--nn"
        help = "score the neural baselines and write their per_class_test.csv"
        arg_type = Bool
        default = false
        "--nn_batch_size"
        arg_type = Int
        default = 128
        "--resize_to"
        arg_type = Int
        default = 224
        "--boost_round"
        arg_type = Int
        default = 0
        "--metric"
        help = "which head selection to read (macro_f1, auroc, ece or bacc)"
        arg_type = String
        default = "macro_f1"
        "--e2e_trial"
        arg_type = String
        default = "MAGE_ALONE_20H"
        "--class_names"
        help = "optional comma-separated class names, in label order"
        arg_type = String
        default = ""
        "--save_name"
        arg_type = String
        default = ""
    end
    return s
end

"""Per-class true positive and true negative rates, in label order."""
function per_class_rates(labels::AbstractVector{<:Integer}, preds::AbstractVector{<:Integer}, n_classes::Int)
    lvls = collect(0:(n_classes - 1))
    y = Int.(collect(labels))
    ŷ = Int.(collect(preds))
    sens = MulticlassTruePositiveRate(average = NoAvg(), return_type = Vector, levels = lvls)(ŷ, y)
    spec = MulticlassTrueNegativeRate(average = NoAvg(), return_type = Vector, levels = lvls)(ŷ, y)
    return Float64.(sens), Float64.(spec)
end

"""Score every neural baseline on the test split and write its per-class table."""
function write_nn_per_class(args)
    @info "Scoring the neural baselines through python_utils.medmnist_resnet_baseline"
    home = dirname(@__DIR__)
    sys = pyimport("sys")
    (home in [pyconvert(String, p) for p in sys.path]) || sys.path.insert(0, home)
    baseline = pyimport("python_utils.medmnist_resnet_baseline")
    np = pyimport("numpy")

    for backbone in ("resnet18", "resnet34", "resnext50")
        experiment = "$(backbone)_$(args["dataset"])"
        out_dir = joinpath(args["output_dir"], "NN_baselines")
        config = baseline.DatasetConfig(
            dataset_name = args["dataset"],
            train_file = joinpath(home, "datasets_pickle", "$(args["dataset"])_64_train.jld2"),
            val_file = joinpath(home, "datasets_pickle", "$(args["dataset"])_64_val.jld2"),
            test_file = joinpath(home, "datasets_pickle", "$(args["dataset"])_64_test.jld2"),
            num_classes = args["num_classes"],
            batch_size = args["nn_batch_size"],
            val_batch_size = args["nn_batch_size"],
            epochs = 1,
            lr = 1e-4,
            weight_decay = 1e-4,
            use_class_weights = true,
            seed = 123,
            resize_to = args["resize_to"],
            output_dir = out_dir,
            experiment_name = experiment,
            backbone_name = backbone,
            nd = args["nd"],
            pretrained_weights = "default",
        )
        # clean_existing = false keeps the trained checkpoints in place
        context = baseline.prepare_experiment(config, clean_existing = false)
        model = baseline.build_model(backbone, args["num_classes"], pretrained_weights = "default").to(context.device)
        checkpoint_path, _ = baseline.select_best_checkpoint(context, "val_balanced_accuracy")
        baseline.load_checkpoint(model, checkpoint_path, context.device)

        _, labels_py, preds_py, _ = baseline.evaluate_loader(model, context.test_dataloader, context.device, args["num_classes"])
        labels = pyconvert(Vector{Int}, np.asarray(labels_py).astype("int64").tolist())
        preds = pyconvert(Vector{Int}, np.asarray(preds_py).astype("int64").tolist())
        sens, spec = per_class_rates(labels, preds, args["num_classes"])

        path = joinpath(out_dir, experiment, "per_class_test.csv")
        mkpath(dirname(path))
        CSV.write(path, DataFrame(
            class = 0:(args["num_classes"] - 1),
            sensitivity = sens,
            specificity = spec,
            support = [count(==(c), labels) for c in 0:(args["num_classes"] - 1)],
        ))
        @info "Wrote neural per-class table" experiment path
    end
end

parse_per_class(field) = [parse(Float64, x) for x in split(strip(String(field)), ';') if !isempty(strip(x))]

"""Per-class rates of every head of one dictionary, as computed by src/train_ml.jl."""
function distilled_columns(root::String, metric::String, suffix::String)
    path = joinpath(root, "extended_metrics_$(metric)_$(suffix).csv")
    isfile(path) || (@warn "Missing distilled metrics" path; return Dict{String, Any}())
    df = CSV.read(path, DataFrame)
    out = Dict{String, Any}()
    for row in eachrow(df)
        row.split == "test" || continue
        out[String(row.model)] = (
            sensitivity = parse_per_class(row.sensitivity),
            specificity = parse_per_class(row.specificity),
        )
    end
    return out
end

function tidy_column(path::String)
    isfile(path) || (@warn "Missing per-class file" path; return nothing)
    df = CSV.read(path, DataFrame)
    sort!(df, :class)
    return (sensitivity = Float64.(df.sensitivity), specificity = Float64.(df.specificity))
end

fmt(x) = isnan(x) ? "--" : @sprintf("%.1f", 100 * x)
cell(col, c) = isnothing(col) ? "--" : string(fmt(col.sensitivity[c]), "/", fmt(col.specificity[c]))

function main()
    args = parse_args(build_parser())
    args["nn"] && write_nn_per_class(args)

    out_dir = args["output_dir"]
    root = joinpath(out_dir, args["trial_id"], "ml_models_boost_$(args["boost_round"])")
    dicts = [("beststrue_false", "Best per dim"), ("bestsfalse_false", "Runs per dim"), ("bestsfalse_true", "All per dim")]
    heads = ["RF", "SVC", "LR"]

    columns = Vector{Pair{String, Any}}()
    # the reader tags its outputs with the --all setting; a single-individual read is enough here,
    # since only the validation-selected program is reported
    e2e_dir = joinpath(out_dir, args["e2e_trial"], "mage_imgcls")
    e2e_candidates = [joinpath(e2e_dir, "per_class_test_allfalse.csv"), joinpath(e2e_dir, "per_class_test_alltrue.csv")]
    e2e_found = filter(isfile, e2e_candidates)
    e2e_path = isempty(e2e_found) ? first(e2e_candidates) : first(e2e_found)
    push!(columns, "MAGE end-to-end" => tidy_column(e2e_path))
    for (suffix, label) in dicts
        cols = distilled_columns(root, args["metric"], suffix)
        for h in heads
            push!(columns, "$label $h" => get(cols, h, nothing))
        end
    end
    for backbone in ("resnet18", "resnet34", "resnext50")
        push!(columns, backbone => tidy_column(joinpath(out_dir, "NN_baselines", "$(backbone)_$(args["dataset"])", "per_class_test.csv")))
    end

    n_classes = args["num_classes"]
    names = isempty(args["class_names"]) ? ["Class $(c - 1)" for c in 1:n_classes] : String.(split(args["class_names"], ','))
    length(names) == n_classes || error("Got $(length(names)) class names for $n_classes classes")

    tidy = DataFrame(class = String[], method = String[], sensitivity = Float64[], specificity = Float64[])
    for c in 1:n_classes, (label, col) in columns
        isnothing(col) && continue
        push!(tidy, (names[c], label, 100 * col.sensitivity[c], 100 * col.specificity[c]))
    end

    save_name = isempty(args["save_name"]) ? joinpath(root, "per_class_sens_spec_$(args["metric"]).csv") : args["save_name"]
    mkpath(dirname(save_name))
    CSV.write(save_name, tidy)
    @info "Wrote tidy per-class metrics" save_name rows = nrow(tidy)

    tex_path = replace(save_name, r"\.csv$" => ".tex")
    open(tex_path, "w") do io
        println(io, "% per-class sensitivity/specificity (\\%) on the test split, head selection by $(args["metric"])")
        println(io, "\\begin{tabular}{l", repeat(" c", length(columns)), "}")
        println(io, "\\toprule")
        println(io, "Class & ", join(["\\rotatebox{90}{$(label)}" for (label, _) in columns], " & "), " \\\\")
        println(io, "\\midrule")
        for c in 1:n_classes
            println(io, names[c], " & ", join([cell(col, c) for (_, col) in columns], " & "), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    @info "Wrote LaTeX table" tex_path

    missing_cols = [label for (label, col) in columns if isnothing(col)]
    isempty(missing_cols) || @warn "Columns without a per-class source" missing_cols
    println("\nSensitivity/specificity (%), test split")
    print(rpad("Class", 16))
    for (label, _) in columns
        print(lpad(first(label, 11), 13))
    end
    println()
    for c in 1:n_classes
        print(rpad(names[c], 16))
        for (_, col) in columns
            print(lpad(cell(col, c), 13))
        end
        println()
    end
end

main()
