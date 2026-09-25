using Random

# ################### #
# AML DATASET ------- -
# ################### #

classes_dictionary_org = Dict(
    "BAS" => 1,
    "EBO" => 2, "EOS" => 3, "KSC" => 4, "LYA" => 5,
    "LYT" => 6, "MMZ" => 7, "MOB" => 8, "MON" => 9,
    "MYB" => 10, "MYO" => 11, "NGB" => 12, "NGS" => 13,
    "PMB" => 14, "PMO" => 15
)

abbreviation_dict = Dict(
    "NGS" => "Neutrophil (segmented)",
    "NGB" => "Neutrophil (band)",
    "EOS" => "Eosinophil",
    "BAS" => "Basophil",
    "MON" => "Monocyte",
    "LYT" => "Lymphocyte (typical)",
    "LYA" => "Lymphocyte (atypical)",
    "KSC" => "Smudge Cell",
    "MYO" => "Myeloblast",
    "PMO" => "Promyelocyte",
    "MYB" => "Myelocyte",
    "MMZ" => "Metamyelocyte",
    "MOB" => "Monoblast",
    "EBO" => "Erythroblast",
    "PMB" => "Promyelocyte (bilobed)"
);

# Binary splits that one might consider
const BLASTS::Vector{String} = ["MOB", "MYO"]
const ATYPICAL::Vector{String} = ["LYA", "EBO", "MOB", "MYO", "MYB", "MMZ", "PMO"]
const NEUTROPHIL_VS_ALL::Vector{String} = ["NGS", "NGB"] # 8,484 + 109
const MONOCYTE_VS_ALL::Vector{String} = ["MON"] # 1,789
const PROMYELOCYTE_VS_ALL::Vector{String} = ["PMO", "PMB"] # 70 + 18
const IMMATURE_VS_ALL::Vector{String} = ["EBO", "MOB", "MYO", "PMO", "PMB", "MYB", "MMZ", "KSC"]

# LABEL BINARIZERS #######################

"""
Splits Blast vs Non blasts based on abbreviation.
"""
function _to_binary_blast_label(l::String)
    if l in BLASTS
        return "BLAST"
    end
    return "NON-BLASTS"
end

"""
Splits Blast vs Non blasts based on abbreviation.
"""
function _to_binary_atypical_label(l::String)
    if l in ATYPICAL
        return "ATYPICAL"
    end
    return "TYPICAL"
end

"""
Splits between Neutrophil (segmented and band) and the rest
based on abbreviation.
"""
function _to_binary_neutrophil_label(l::String)
    if l in NEUTROPHIL_VS_ALL
        return "NEUTROPHIL"
    end
    return "NON-NEUTROPHIL"
end

function _to_binary_monocyte_label(l::String)
    if l in MONOCYTE_VS_ALL
        return "MONOCYTE"
    end
    return "NON-MONOCYTE"
end

function _to_binary_promyelocyte_label(l::String)
    if l in PROMYELOCYTE_VS_ALL
        return "PROMYELOCYTE"
    end
    return "NON-PROMYELOCYTE"
end

function _to_binary_immature_label(l::String)
    if l in IMMATURE_VS_ALL
        return "IMMATURE"
    end
    return "NON-IMMATURE"
end

# LABEL IDENTITY #######################
function _to_identity_label(l::String)
    return classes_dictionary_org[l] # returns an index for that class 15 possibilities.
end

#########
# READ  #
#########

"""
Holds an image and it's channels.

The label is one of the abbreviations from `abbreviation_dict`
"""
struct aml_file
    path::String
    label::String
    target::Int
    binary_target::Int
    img
    channels

    function aml_file(dataset_base_path, path, label_mapper, label_binarizer; gray::Bool = false, hsv::Bool = false, hed::Bool = false, resize_to::Union{Tuple{Int, Int}, Nothing} = nothing)
        p = joinpath(dataset_base_path, path)
        img = load(joinpath(dataset_base_path, path))
        color_img = TiffImages.color.(img)
        channels = _process_image_channels(color_img, gray, hsv, hed; resize_to = resize_to)
        true_label_abbreviation = String(split(path, "/")[1])
        ml_label = label_mapper(true_label_abbreviation) # real class
        ml_label_binary = label_binarizer(true_label_abbreviation) # grouped class or identity

        return new(
            p, true_label_abbreviation,
            ml_label,
            ml_label_binary,
            color_img, channels
        )
    end
end

"""
Get's a human readable name from the label abbreviation 
"""
function aml_file_readable_name(f::aml_file)
    global abbreviation_dict
    return abbreviation_dict[f.label]
end

"""
Gathers multiple aml_files
"""
struct aml_files
    files::Vector{aml_file}
    function aml_files(annotations::DataFrame, label_mapper, label_binarizer, dataset_base_path, ; gray::Bool = false, hsv::Bool = false, hed::Bool = false, resize_to::Union{Tuple{Int, Int}, Nothing} = nothing)
        names = annotations[:, 1]
        l = length(names)
        files = Vector{aml_file}(undef, l)
        @info "Reading $l files"
        itr = collect(enumerate(names))
        @threads for (idx, name) in itr
            files[idx] = aml_file(dataset_base_path, name, label_mapper, label_binarizer; gray, hsv, hed, resize_to)
        end
        return new(files)
    end
end

Base.size(a::aml_files) = Base.size(a.files)
Base.length(a::aml_files) = Base.length(a.files)
Base.getindex(A::aml_files, i::Int) = Base.getindex(A.files, i)
Base.getindex(A::aml_files, I::Vararg{Int, N}) where {N} = Base.getindex(A.files, I)
Base.getindex(A::aml_files, VI::Vector{Int}) where {N} = Base.getindex(A.files, VI)
Base.iterate(A::aml_files, state = 1) =
    state > Base.length(A) ? nothing : (A.files[state], state + 1)
MLUtils.numobs(A::aml_files) = length(A)

get_label(f::aml_file) = f.label
get_ml_target(f::aml_file) = f.target
get_ml_target_binary(f::aml_file) = f.binary_target
get_color_channels(f::aml_file) = f.channels
get_path(f::aml_file) = f.path


# # AML
function load_aml_dataset(
        dataset_base_path::String,
        annot_base_path::String;
        label_mapper,
        label_binarizer::Union{Function, Nothing} = nothing,
        percentages = (0.2, 0.2),
        gray::Bool = false, hsv::Bool = false, hed::Bool = false, resize_to::Union{Tuple{Int, Int}, Nothing} = nothing, seed::Int = 42
    )
    rng = Xoshiro(seed)
    isnothing(label_binarizer) ? label_binarizer = label_mapper : nothing # if the binarizer is nothing get_ml_target_binary == get_ml_target
    extract_target_label = isnothing(label_binarizer) ? get_ml_target : get_ml_target_binary
    annotations = DataFrame(
        readdlm(
            joinpath(dataset_base_path, annot_base_path, "annotations.dat")
        ),
        ["path", "annot1", "annot2", "annot3"]
    )

    files = aml_files(annotations, label_mapper, label_binarizer, dataset_base_path; gray, hsv, hed, resize_to)
    Y = extract_target_label.(files) # either binary or multi class depends on whether label_binarizer was passed
    val, test, train = splitobs(rng, (collect(1:length(files)), Y), at = percentages, shuffle = true, stratified = Y)

    train_files = files[collect(train[1])]
    val_files = files[collect(val[1])]
    test_files = files[collect(test[1])]

    trainx = get_color_channels.(train_files)
    valx = get_color_channels.(val_files)
    testx = get_color_channels.(test_files)

    trainy = train[2]
    valy = val[2]
    testy = test[2]

    # sanity checks
    @assert length(trainx) == length(trainy)
    @assert length(valx) == length(valy)
    @assert length(testx) == length(testy)

    # all indices are in train,val or test
    tmp = []
    append!(tmp, train[1])
    append!(tmp, val[1])
    append!(tmp, test[1])
    @assert Set(tmp) == Set(collect(1:length(files)))

    allx, ally = [trainx..., valx...], [trainy..., valy...]

    tmp = countmap(trainy)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Train")
    tmp = countmap(valy)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Validation")
    tmp = countmap(testy)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Test")
    return (
        identity.(trainx), identity.(trainy), identity.(valx), identity.(valy), identity.(testx), identity.(testy), allx, ally, Dict(
            :train_indices => train[1],
            :val_indices => val[1],
            :test_indices => test[1],
            :label_mapper => classes_dictionary_org,
            :abbreviations => abbreviation_dict,

            # paths
            :train_paths => get_path.(train_files),
            :val_paths => get_path.(val_files),
            :test_paths => get_path.(test_files),
        ),
    )

    # Add constants
    # constants = [0.0, -1.0, 0.5, 2.0, 10, 20.0, 30.0]
    # for container in (trainx, valx, testx)
    #     for obs in container
    #         push!(obs, constants...)
    #     end
    # end
end
