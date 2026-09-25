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
    "--use_only_bests"
    arg_type = Bool #
    default = true
    "--use_really_all"
    arg_type = Bool #
    default = false
    "--multiple_of"
    arg_type = Int
    default = 10
    "--output_dir"
    arg_type = String
    "--use_ski"
    arg_type = Bool
    default = false
    "--act_family"
    arg_type = String
    default = "all"
    "--boost_round"
    arg_type = Int
    "--metric_of_interest"
    arg_type = String
    default = "nbacc"
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
    "--single_grade_files"
    arg_type = String
    default = ""
    "--composition_grade_files"
    arg_type = String
    default = ""
    "--rf_n_estimators"
    arg_type = Int
    default = 200
    "--rf_max_depth"
    arg_type = Int
    default = 20
    "--rf_min_samples_split"
    arg_type = Int
    default = 10
    "--rf_min_samples_leaf"
    arg_type = Int
    default = 2
    "--rf_max_features"
    arg_type = Float64
    default = 0.5
    "--rf_max_samples"
    arg_type = Float64
    default = 0.7
    "--rf_criterion"
    arg_type = String
    default = "gini"
end
rootdir = "./"
Parsed_args = parse_args(s)
@show Parsed_args

METRIC_OF_INTEREST = Parsed_args["metric_of_interest"]
BOOST_ROUND = Parsed_args["boost_round"]
ACT_FAMILY = Parsed_args["act_family"]
@assert ACT_FAMILY in ["all", "regression", "classification"] "act_family must be one of all, regression, classification"
include(joinpath(home, "src", "mage_imports.jl"))
include(joinpath(home, "src", "magenet_ski.jl"))

USE_SKI = Parsed_args["use_ski"]
USE_SKI ? addprocs(nt, exeflags = ["--threads=1"]) : nothing
@everywhere using UTCGP, MAGE_SKIMAGE_MEASURE, Images

include(joinpath(home, "utils", "utils.jl"))
include(joinpath(home, "utils", "utils_aml.jl"))
include(joinpath(home, "utils", "datasets.jl"))
include(joinpath(home, "utils", "activations.jl"))
include(joinpath(home, "src", "interpretability", "composition_keys.jl"))

use_bests = Parsed_args["use_only_bests"]
use_really_all = Parsed_args["use_really_all"]
multiple_of = Parsed_args["multiple_of"]
Suffix = "$(use_bests)_$(use_really_all)"
@info "Activation export family" ACT_FAMILY

@assert multiple_of > 0 "--multiple_of must be > 0"

function checkpoint_number(filename::String)
    m = match(r"^checkpoint_(\d+)\.pickle$", filename)
    return isnothing(m) ? nothing : parse(Int, m.captures[1])
end

# Read data ---
data_path = Parsed_args["data_location"]
data = load(data_path)["single_stored_object"]
trainx, trainy = data.xs, data.ys
trainx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in trainx]

val_location = Parsed_args["val_data_location"]
test_location = Parsed_args["test_data_location"]
has_val_data = val_location != ""
has_test_data = test_location != ""
if has_val_data
    data = load(val_location)["single_stored_object"]
    valx, valy = data.xs, data.ys
    valx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in valx]
    @assert length(valx) == length(valy)
else
    valx, valy = nothing, nothing
    VALDataloader = nothing
end

testx, testy = nothing, nothing
if has_test_data
    data = load(test_location)["single_stored_object"]
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
folders_to_read = filter(x -> occursin(r"[0-9]_[0-9]*", x), readdir(folder))
folders_to_read = [joinpath(folder, x) for x in folders_to_read]

# filter already computed ones
all_previously_read = String[]
if BOOST_ROUND != 0
    for prev_round_idx in 0:(BOOST_ROUND - 1)
        read_in_round = readlines(
            joinpath(
                folder,
                "best_modules_boost_$(prev_round_idx)/modules_read.txt"
            )
        )
        append!(all_previously_read, read_in_round)
    end
end
@show all_previously_read

folders_to_read = filter(x -> !(x in all_previously_read), folders_to_read)
@info "Folders to read after filtering : $folders_to_read"

Bests = Dict()
Files_Losses = Dict()
Module_Losses = Dict()
BestPerModule = Dict{Tuple{Int, Int}, NamedTuple}()
not_read = []
for folder_to_read in folders_to_read
    @info "Reading $folder_to_read"
    best_losses = []
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
            to_omit = 0
            if haskey(JSON.parse(metrics_file_content[begin]), "params")
                println("Params at FIRST line")
                metrics = metrics_file_content[begin]
                to_omit = 1
            elseif haskey(JSON.parse(metrics_file_content[end]), "params")
                println("Params at LAST line")
                metrics = metrics_file_content[end]
                to_omit = length(metrics_file_content)
            else
                @warn "NO METRICS IN FILE ? "
            end

            # GET BEST VALUE OF RUN
            vs = []
            for (i, l) in enumerate(metrics_file_content)
                try
                    if i == to_omit
                        continue
                    end
                    push!(vs, JSON.parse(l)[METRIC_OF_INTEREST])
                catch e
                    @warn "Could not read $l : $e"
                end
            end
            best_f = minimum(vs)
            @info "Best metric for $(metrics_file) : $best_f"
            push!(
                best_losses,
                (loss = best_f, root = root, metrics_file = metrics_file)
            )
        catch e
            @show e
            @warn root
        end
    end
    sort!(best_losses, by = first)
    @show best_losses
    Module_Losses[folder_to_read] = [pack.loss for pack in best_losses]
    best = best_losses[begin]
    best_ind_path = joinpath(best.root, "checkpoint_0.pickle")
    best_ind = deserialize(best_ind_path)["best_genome"]
    best_key_match = match(r"(?<=\/)[0-9]_[0-9]*", best.root)
    if !isnothing(best_key_match)
        ktxt = best_key_match.match
        Kbest = (parse(Int, ktxt[1]), parse(Int, ktxt[3:end]))
        BestPerModule[Kbest] = (
            module_key = ktxt,
            loss = best.loss,
            path = best_ind_path,
            root = best.root,
            ind = best_ind,
        )
    end

    if use_bests
        if length(best_losses) > 1
            others_min = map(x -> x.loss, best_losses[(begin + 1):end]) |> minimum
            others_mean = map(x -> x.loss, best_losses[(begin + 1):end]) |> mean
            @info """Best ind in $folder_to_read found in $(best.root).
            best loss : $(best.loss). Others losses (min, mean): $others_min, $others_mean
            """
        else
            @info """Best ind in $folder_to_read found in $(best.root).
            best loss : $(best.loss).
            """
        end
        # read ind
        ind_path = best_ind_path
        @show best.root
        k = match(r"(?<=\/)[0-9]_[0-9]*", best.root).match
        @show k
        K = (parse(Int, k[1]), parse(Int, k[3:end]))
        Files_Losses[best.root] = best.loss
        Bests[K] = (loss = best.loss, path = ind_path, root = best.root, ind = best_ind)
    else
        # reading all
        @info """Best ind in $folder_to_read found in $(best.root).
        best loss : $(best.loss). Reading all.
        """
        for ind_info in best_losses
            if use_really_all
                all_files = filter(x -> occursin("checkpoint", x), readdir(ind_info.root))
                @assert "checkpoint_0.pickle" in all_files
                filter!(all_files) do file
                    num = checkpoint_number(file)
                    !isnothing(num) && (num == 0 || num % multiple_of == 0)
                end
                sort!(all_files, by = x -> checkpoint_number(x))
                @warn "This run had this number of checkpoints $(length(all_files)). First five : $(all_files[1:5])"
                for file in all_files
                    ind_path = joinpath(ind_info.root, file)
                    best_ind = deserialize(ind_path)["best_genome"]
                    k = match(r"(?<=\/)[0-9]_[0-9]*", best.root).match
                    K = (parse(Int, k[1]), parse(Int, k[3:end]))
                    place = get!(Bests, K, [])
                    push!(place, (loss = ind_info.loss, best_loss = best.loss, path = ind_path, root = best.root, ind = best_ind))
                end
            else
                ind_path = joinpath(ind_info.root, "checkpoint_0.pickle")
                best_ind = deserialize(ind_path)["best_genome"]
                k = match(r"(?<=\/)[0-9]_[0-9]*", best.root).match
                K = (parse(Int, k[1]), parse(Int, k[3:end]))
                place = get!(Bests, K, [])
                push!(place, (loss = ind_info.loss, best_loss = best.loss, path = ind_path, root = best.root, ind = best_ind))
            end
            # ind_path = joinpath(ind_info.root, "checkpoint_0.pickle")
            # best_ind = deserialize(ind_path)["best_genome"]
            # k = match(r"[0-9]_[0-9]*", best.root).match
            # K = (parse(Int, k[1]), parse(Int, k[3:end]))
            # place = get!(Bests, K, [])
            # push!(place, (loss = best.loss, path = ind_path, root = best.root, ind = best_ind))
        end
    end
end
@show not_read

# Extract Inds :
ks = keys(Bests) |> collect |> sort
for (k, v) in Module_Losses
    @info "In folder $k. Best losses per seed : $(v)"
end

function module_sort_key(module_key::AbstractString)
    m = match(r"^(\d+)_(\d+)$", module_key)
    @assert !isnothing(m) "Invalid module key format: $module_key"
    return (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

function save_runs_scores_csv(folder::String, module_losses::AbstractDict)
    module_keys = String[]
    keyed_losses = Pair{String, Vector{Float64}}[]
    for (path, losses) in module_losses
        @assert path isa AbstractString "Expected module loss key to be a path-like string, got $(typeof(path))"
        @assert losses isa AbstractVector "Expected module losses to be a vector, got $(typeof(losses))"
        module_match = match(r"(?<=/)[0-9]_[0-9]*$", path)
        @assert !isnothing(module_match) "Could not extract module key from path: $path"
        module_key = module_match.match
        push!(module_keys, module_key)
        push!(keyed_losses, module_key => Float64[Float64(loss) for loss in losses])
    end
    sort!(module_keys, by = module_sort_key)

    losses_by_key = Dict(keyed_losses)
    n_rows = isempty(module_keys) ? 0 : maximum(length(losses_by_key[key]) for key in module_keys)
    cols = Pair{String, Vector{Union{Missing, Float64}}}[]
    for key in module_keys
        losses = losses_by_key[key]
        padded = Vector{Union{Missing, Float64}}(missing, n_rows)
        for (i, loss) in enumerate(losses)
            padded[i] = loss
        end
        push!(cols, key => padded)
    end

    scores_df = DataFrame(cols)
    out_dir = joinpath(folder, "best_surrogates_$(BOOST_ROUND)")
    isdir(out_dir) || mkpath(out_dir)
    out_path = joinpath(out_dir, "runs_scores.csv")
    CSV.write(out_path, scores_df)
    @info "Saved runs scores table" out_path n_rows n_cols = length(module_keys)
    return out_path
end

save_runs_scores_csv(folder, Module_Losses)

if use_bests
    pop_entries = [Bests[k] for k in ks]
    pop = [entry.ind for entry in pop_entries]
else
    pop_entries = reduce(vcat, [Bests[k] for k in ks])
    pop = map(i -> i.ind, pop_entries)
    @info "Pop length : $(length(pop))"
end

N_NODES = pop[1][1] |> length
node_config = nodeConfig(N_NODES, 1, 3, n_ins)
shared_in, _ = make_evolvable_utgenome(
    model_arch, ml, node_config
)

decoded_pop_programs = [UTCGP.decode_with_output_nodes(ind, ml, model_arch, shared_in) for ind in pop]
program_key_features = Dict{Tuple{Int, Int}, Any}()
for (program_idx, decoded_programs) in enumerate(decoded_pop_programs)
    for output_idx in 1:length(decoded_programs)
        program_key_features[(program_idx, output_idx)] = extract_program_keys(decoded_programs[output_idx])
    end
end

# TRAIN DATA
nt = Threads.nthreads()

# Save best modules
UTCGP.reset_genome!.(pop)
folder_pop = joinpath(folder, "best_modules_boost_$BOOST_ROUND")
isdir(folder_pop) || mkdir(folder_pop)
open(joinpath(folder_pop, "best_modules_bests$(Suffix).pickle"), "w") do io
    write(io, UTCGP.general_serializer(pop))
end
open(joinpath(folder_pop, "modules_read.txt"), "w") do io
    for i in folders_to_read
        println(io, i)
    end
end

function save_best_programs_code_per_module!(
        folder_pop::String,
        best_per_module::Dict{Tuple{Int, Int}, NamedTuple},
        ml,
        model_arch,
        shared_in,
    )
    out_dir = joinpath(folder_pop, "best_progs_code")
    isdir(out_dir) || mkpath(out_dir)
    for k in sort(collect(keys(best_per_module)))
        entry = best_per_module[k]
        decoded = UTCGP.decode_with_output_nodes(entry.ind, ml, model_arch, shared_in)
        lines = String[
            "# module_key=$(entry.module_key)",
            "# checkpoint_path=$(entry.path)",
            "# best_subproblem_loss=$(entry.loss)",
            "",
        ]
        for (output_idx, program) in enumerate(decoded)
            seq = UTCGP.compile_program(program, model_arch, ml; safe = true)
            push!(lines, "# output_idx=$(output_idx)")
            push!(lines, UTCGP.sequential_source(seq))
            push!(lines, "")
        end
        write(joinpath(out_dir, "$(entry.module_key).jl"), join(lines, "\n"))
    end
    @info "Saved best module program code" out_dir n_modules = length(best_per_module)
    return out_dir
end

save_best_programs_code_per_module!(folder_pop, BestPerModule, ml, model_arch, shared_in)

function raw_variant_dir(folder_pop::String, act_name::String)
    return joinpath(folder_pop, act_name)
end

function save_activation_header(folder_pop::String, act_name::String, suffix::String, header_rows)
    out_dir = raw_variant_dir(folder_pop, act_name)
    isdir(out_dir) || mkpath(out_dir)
    header_df = DataFrame(header_rows)
    header_path = joinpath(out_dir, "best_modules_header_bests$(suffix).csv")
    CSV.write(header_path, header_df)
    return header_path
end

function save_activation_dataset(folder_pop::String, split::String, act_name::String, preds::Matrix, gts, datax)
    out_dir = raw_variant_dir(folder_pop, act_name)
    isdir(out_dir) || mkpath(out_dir)
    @assert size(preds, 1) == length(gts) "Prediction rows must match GT length"
    initial_x = map(z -> map(i -> reinterpret.(UInt8, i.img), z), datax)
    payload = (
        ys = [preds[i, :] for i in 1:size(preds, 1)],
        gt = gts,
        extras = (
            sruct_ = CLASSIFICATION_DATASET_VEC_SCALAR,
            initial_x = initial_x,
            act = act_name,
        ),
    )
    b = "bests$(Suffix)"
    CSV.write(joinpath(out_dir, "$(split)_mat_$b.csv"), DataFrame(preds, :auto))
    CSV.write(joinpath(out_dir, "$(split)y_$b.csv"), DataFrame(y = gts))
    save_path = joinpath(out_dir, "best_modules_dataset_$(split)_$b.jld2")
    @info "Saving activation dataset" split act_name save_path n_samples = size(preds, 1) n_programs = size(preds, 2)
    return save_object(save_path, payload)
end

function activation_output_columns(activated_preds)
    first_pred = activated_preds[1]
    if first_pred isa AbstractVector
        n_outputs = length(first_pred)
        return [[pred[i] for pred in activated_preds] for i in 1:n_outputs]
    end
    return [activated_preds]
end

function _parse_grade_file_list(raw::String)
    txt = strip(raw)
    isempty(txt) && return String[]
    return String[strip(s) for s in split(txt, ",") if !isempty(strip(s))]
end

function _persona_from_grade_file(path::String; composition::Bool)
    stem = basename(path)
    stem = replace(stem, r"\.(md|json)$" => "")
    stem = replace(stem, r"_grade$" => "")
    if composition
        stem = replace(stem, "_composition" => "")
        stem = replace(stem, "_compo" => "")
        stem = strip(stem, '_')
    end
    return stem
end

function _parse_library_id(raw)
    if raw isa Integer
        return Int(raw)
    end
    s = strip(String(raw))
    m = match(r"(?i)(\d+)", s)
    @assert !isnothing(m) "Could not parse library_id from $raw"
    return parse(Int, m.captures[1])
end

function _iter_grade_payloads(parsed)
    if parsed isa AbstractVector
        return parsed
    end
    return Any[parsed]
end

function _load_grade_scores(paths::Vector{String}; composition::Bool)
    per_persona = Dict{String, Dict{Tuple{Int, String}, Float64}}()
    for rel in paths
        abs_path = rel
        @assert isfile(abs_path) "Grade file not found: $abs_path"
        persona = _persona_from_grade_file(rel; composition = composition)

        score_lists = Dict{Tuple{Int, String}, Vector{Float64}}()
        parsed = JSON.parsefile(abs_path)
        for payload in _iter_grade_payloads(parsed)
            @assert haskey(payload, "library_id") "Missing library_id in $abs_path"
            @assert haskey(payload, "scores") "Missing scores in $abs_path"
            current_library = _parse_library_id(payload["library_id"])
            for entry in payload["scores"]
                @assert haskey(entry, "unit") "Missing unit in one score entry from $abs_path"
                @assert haskey(entry, "score") "Missing score in one score entry from $abs_path"
                unit = strip(String(entry["unit"]))
                isempty(unit) && continue
                score = Float64(entry["score"])
                key = (current_library, unit)
                push!(get!(score_lists, key, Float64[]), score)
            end
        end
        @assert !isempty(score_lists) "No scores could be parsed from $abs_path"
        scores = Dict{Tuple{Int, String}, Float64}()
        for (k, vals) in score_lists
            scores[k] = mean(vals)
        end
        per_persona[persona] = scores
        @info "Loaded grade file" persona composition n_scores = length(scores) path = abs_path
    end
    return per_persona
end

function _fit_rf_and_importances(train_mat::Matrix{Float64}, train_labels::Vector{Int})
    sklearn_ensemble = pyimport("sklearn.ensemble")
    rf = sklearn_ensemble.RandomForestClassifier(
        n_estimators = Parsed_args["rf_n_estimators"],
        max_depth = Parsed_args["rf_max_depth"],
        min_samples_split = Parsed_args["rf_min_samples_split"],
        min_samples_leaf = Parsed_args["rf_min_samples_leaf"],
        max_features = Parsed_args["rf_max_features"],
        max_samples = Parsed_args["rf_max_samples"],
        criterion = Parsed_args["rf_criterion"],
        class_weight = "balanced",
        n_jobs = -1,
        random_state = 1,
    )
    rf.fit(train_mat, train_labels)
    return pyconvert(Vector{Float64}, rf.feature_importances_)
end

function _score_program_keys(step_keys_by_lib::Dict{Int, Vector{String}}, persona_scores::Dict{Tuple{Int, String}, Float64}, score_kind::String, persona::String)
    vals = Float64[]
    for (lib, keys) in step_keys_by_lib
        for key in keys
            lk = (lib, key)
            haskey(persona_scores, lk) || error("Missing $(score_kind) score for persona=$(persona), library=$(lib), key=$(key)")
            push!(vals, persona_scores[lk])
        end
    end
    if isempty(vals)
        return (mean = NaN, min = NaN)
    end
    return (mean = mean(vals), min = minimum(vals))
end

function _build_and_save_program_stats_table(
        folder_pop::String,
        act_name::String,
        suffix::String,
        header_rows::Vector{<:NamedTuple},
        train_preds::Matrix{Float64},
        train_labels::Vector{Int},
        program_key_features::Dict{Tuple{Int, Int}, Any},
        single_personas::Vector{String},
        composition_personas::Vector{String},
        single_score_tables::Dict{String, Dict{Tuple{Int, String}, Float64}},
        composition_score_tables::Dict{String, Dict{Tuple{Int, String}, Float64}},
    )
    importances = _fit_rf_and_importances(train_preds, train_labels)
    @assert length(importances) == length(header_rows) "RF importances must align with header rows."

    rows = NamedTuple[]
    for row in header_rows
        selected_program_idx = Int(row.selected_program_idx)
        output_idx = Int(row.output_idx)
        key = (selected_program_idx, output_idx)
        haskey(program_key_features, key) || error("Program key features missing for selected_program_idx=$(selected_program_idx), output_idx=$(output_idx)")
        feats = program_key_features[key]

        base = OrderedDict{Symbol, Any}()
        base[:checkpoint_path] = row.checkpoint_path
        base[:checkpoint_name] = row.checkpoint_name
        base[:module_key] = row.module_key
        base[:n_fn_lib1] = get(feats.n_fn_by_lib, 1, 0)
        base[:n_fn_lib2] = get(feats.n_fn_by_lib, 2, 0)
        base[:n_fn_lib3] = get(feats.n_fn_by_lib, 3, 0)
        base[:n_fn_lib4] = get(feats.n_fn_by_lib, 4, 0)
        base[:n_steps] = feats.n_steps
        base[:rf_importance_rank] = 0
        base[:rf_importance] = importances[Int(row.column_idx)]
        base[Symbol("BACC(subproblem)")] = Float64(row.subproblem_score)

        for persona in single_personas
            score_table = single_score_tables[persona]
            agg = _score_program_keys(feats.single_steps_by_lib, score_table, "single", persona)
            base[Symbol("min_$(persona)_single")] = agg.min
            base[Symbol("mean_$(persona)_single")] = agg.mean
        end
        for persona in composition_personas
            score_table = composition_score_tables[persona]
            agg = _score_program_keys(feats.composition_steps_by_lib, score_table, "composition", persona)
            base[Symbol("min_$(persona)_composition1")] = agg.min
            base[Symbol("mean_$(persona)_composition1")] = agg.mean
        end
        base[:activation] = row.activation
        base[:output_idx] = output_idx
        base[:seed_root] = row.seed_root
        push!(rows, (; base...))
    end

    df = DataFrame(rows)
    sort!(df, :rf_importance, rev = true)
    df[!, :rf_importance_rank] = collect(1:nrow(df))

    ordered_cols = Symbol[
        :checkpoint_path,
        :checkpoint_name,
        :module_key,
        :n_fn_lib1,
        :n_fn_lib2,
        :n_fn_lib3,
        :n_fn_lib4,
        :n_steps,
        :rf_importance_rank,
        :rf_importance,
        Symbol("BACC(subproblem)"),
    ]
    for persona in single_personas
        push!(ordered_cols, Symbol("min_$(persona)_single"))
    end
    for persona in single_personas
        push!(ordered_cols, Symbol("mean_$(persona)_single"))
    end
    for persona in composition_personas
        push!(ordered_cols, Symbol("min_$(persona)_composition1"))
    end
    for persona in composition_personas
        push!(ordered_cols, Symbol("mean_$(persona)_composition1"))
    end
    push!(ordered_cols, :activation)
    push!(ordered_cols, :output_idx)
    push!(ordered_cols, :seed_root)
    select!(df, ordered_cols)

    out_dir = raw_variant_dir(folder_pop, act_name)
    isdir(out_dir) || mkpath(out_dir)
    out_path = joinpath(out_dir, "best_modules_program_stats_bests$(suffix).csv")
    CSV.write(out_path, df)
    @info "Saved per-program stats table" out_path n_rows = nrow(df) n_cols = ncol(df)
    return out_path
end

single_grade_files = _parse_grade_file_list(Parsed_args["single_grade_files"])
composition_grade_files = _parse_grade_file_list(Parsed_args["composition_grade_files"])
single_score_tables = _load_grade_scores(single_grade_files; composition = false)
composition_score_tables = _load_grade_scores(composition_grade_files; composition = true)
single_personas = String[unique([_persona_from_grade_file(path; composition = false) for path in single_grade_files])...]
composition_personas = String[unique([_persona_from_grade_file(path; composition = true) for path in composition_grade_files])...]
should_compute_program_scores = !isempty(single_grade_files) || !isempty(composition_grade_files)

data = []
push!(data, (trainx, trainy, "train"))
if has_val_data
    push!(data, (valx, valy, "val"))
end
if has_test_data
    push!(data, (testx, testy, "test"))
end

for (datax, datay, split) in data
    @info "Starting split export" split n_samples = length(datax) n_programs = length(pop)
    raw_outs_all = []
    for (i, _ind) in enumerate(pop)
        @info "Running program on split" split program_idx = i total_programs = length(pop)
        xs = datax
        ys = datay
        prog = decoded_pop_programs[i]
        progs = [deepcopy(prog) for i in 1:nt]
        pop_size = length(progs)
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
                    out_v[sample_idx] = (pred = outputs, gt = y)
                end
            end
            push!(tasks, t)
        end
        fetch.(tasks)
        push!(raw_outs_all, deepcopy(OUTS))
    end

    @assert !isempty(raw_outs_all) "Expected at least one program prediction pack"
    @assert length(raw_outs_all[1]) == length(datay) "First program pack must match split length"
    gts = [pack.gt for pack in raw_outs_all[1]]
    @assert gts == datay "GTs recovered from raw predictions must match provided labels"

    first_pred = raw_outs_all[1][1].pred
    n_outputs = first_pred isa AbstractVector ? length(first_pred) : 1
    acts_to_export = compatible_activations_for_output_length(n_outputs; act_family = ACT_FAMILY)
    @assert !isempty(acts_to_export) "No compatible activations found for output length $n_outputs"
    @info "Compatible activations for split" split ACT_FAMILY n_outputs act_names = map(activation_name, acts_to_export)

    for act in acts_to_export
        act_name = activation_name(act)
        @info "Materializing activation outputs" split act_name
        outs_preds = []
        header_rows = NamedTuple[]
        for (program_idx, pack) in enumerate(raw_outs_all)
            activated_preds = [act(sample_pack.pred) for sample_pack in pack]
            act_columns = activation_output_columns(activated_preds)
            append!(outs_preds, act_columns)

            source_entry = pop_entries[program_idx]
            checkpoint_name = basename(source_entry.path)
            module_key_match = match(r"(?<=\/)[0-9]_[0-9]*", source_entry.root)
            module_key = isnothing(module_key_match) ? missing : module_key_match.match
            for output_idx in 1:length(act_columns)
                push!(
                    header_rows,
                    (
                        column_idx = length(header_rows) + 1,
                        module_key = module_key,
                        seed_root = source_entry.root,
                        checkpoint_path = source_entry.path,
                        checkpoint_name = checkpoint_name,
                        selected_program_idx = program_idx,
                        output_idx = output_idx,
                        activation = act_name,
                        subproblem_score = source_entry.loss,
                    ),
                )
            end
        end
        mage_preds = reduce(hcat, outs_preds)
        @assert size(mage_preds, 1) == length(gts) "Activated prediction matrix must match split length"
        @assert size(mage_preds, 2) >= length(pop) "Activated prediction matrix must have at least one column per program"
        @assert size(mage_preds, 2) == length(header_rows) "Header columns must match activation matrix columns"
        header_path = save_activation_header(folder_pop, act_name, Suffix, header_rows)
        @info "Saved activation header" split act_name header_path n_columns = length(header_rows)
        save_activation_dataset(folder_pop, split, act_name, mage_preds, gts, datax)
        if split == "train" && should_compute_program_scores
            _build_and_save_program_stats_table(
                folder_pop,
                act_name,
                Suffix,
                header_rows,
                Float64.(mage_preds),
                Int.(gts),
                program_key_features,
                single_personas,
                composition_personas,
                single_score_tables,
                composition_score_tables,
            )
        end
    end
    @info "Finished split export" split
end
