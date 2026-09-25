using Base: first_index
using FileIO
using Base.Threads
using UTCGP
import PNGFiles
using Images
using StatsBase: sample, Weights
using ThreadPools
using DataFlowTasks
using Logging
using PythonCall
import Flux
using StatisticalMeasures
using CategoricalDistributions
import Statistics
using MLUtils
using Base.Threads
using DataFrames
using DelimitedFiles
using StatsBase: countmap
using Random

# ################### #
# PYTHON HELPER ----- -
# ################### #

abstract type DATASET end

abstract type SURROGATE_DATASET <: DATASET end # comming from NN
abstract type CLASSICAL_DATASET <: DATASET end # normal xs, ys

abstract type REGRESSION_DATASET <: CLASSICAL_DATASET end
abstract type CLASSIFICATION_DATASET <: CLASSICAL_DATASET end

struct REGRESSION_DATASET_IMG_SCALAR{IN_T, OUT_T} <: REGRESSION_DATASET
    xs::Vector{IN_T}
    ys::Vector{OUT_T}
    extras::Dict # true xs for example
end
# struct REGRESSION_DATASET_IMG_SCALARS <: REGRESSION_DATASET end # TODO
# struct REGRESSION_DATASET_VEC_SCALAR <: REGRESSION_DATASET end # TODO

"""
Like MNIST. X = input. Y = class
"""
struct CLASSIFICATION_DATASET_IMG_SCALAR{IN_T, OUT_T} <: CLASSIFICATION_DATASET where {IN_T, OUT_T}
    xs::Vector{IN_T}
    ys::Vector{OUT_T}
    extras::Dict
end

"""
Like MAGE Modules to class. X = features. Y = class
"""
struct CLASSIFICATION_DATASET_VEC_SCALAR{IN_T, OUT_T} <: CLASSIFICATION_DATASET where {IN_T, OUT_T}
    xs::Vector{IN_T}
    ys::Vector{OUT_T}
    extras::Dict
end


"""
Holds data comming from NN module
"""
struct SurrogateDataset_IMG_SCALAR <: SURROGATE_DATASET
    xs::Vector{Vector{<:UTCGP.SImageND}}
    model_preds::Vector{<:AbstractFloat}
    gt::Vector{Int}
    extras::Dict
end

struct SurrogateDataset_IMG_SCALARS <: SURROGATE_DATASET end


# ################### #
# PYTHON HELPER ----- -
# ################### #

pyexec(
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage.color import rgb2hed, hed2rgb, rgb2gray
    import skimage 
    """,
    Main
)

const PYLOCK = ReentrantLock()

function gc_()
    return PythonCall.GIL.lock(PythonCall.GC.gc)
end

function pycall_lock(f::Function)
    lock(PYLOCK)
    try
        local h
        local e
        local d
        t = @tspawnat 1 begin
            h, e, d = PythonCall.GIL.lock(f)
        end
        fetch(t)
        return h, e, d
    finally
        # if rand() > 0.99
        # PythonCall.GIL.lock(PythonCall.GC.gc)
        # end
        unlock(PYLOCK)
    end
end

function rgb2hed(rgbimg)
    h, e, d = pycall_lock() do
        h, e, d = PythonCall.pyexec(
            @NamedTuple{h::Array{Float64, 2}, e::Array{Float64, 2}, d::Array{Float64, 2}}, """
            #import numpy as np
            #import matplotlib.pyplot as plt
            #from skimage import data
            #from skimage.color import rgb2hed, hed2rgb, rgb2gray
            #import skimage 
            #print(skimage.__version__)
            # Example IHC image
            #ihc_rgb = data.immunohistochemistry()

            # Separate the stains from the IHC image
            ihc_hed = rgb2hed(img)

            #print("#############")
            #print(img[0:2,0:2,:])
            #print(img[0:2,0:2,:].shape)

            # Create an RGB image for each of the stains
            null = np.zeros_like(ihc_hed[:, :, 0])
            ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
            ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

            #print(type(ihc_h))

            #print((ihc_hed[:, :, 1])[0:2,0:2])

            #h,e,d =  ihc_h,  ihc_e,  ihc_d

            h,e,d =  rgb2gray(ihc_h),  rgb2gray(ihc_e),  rgb2gray(ihc_d)
            #h,e,d = ihc_hed[:, :, 0], ihc_hed[:, :, 1], ihc_hed[:, :, 2]
            """,
            Main,
            (img = float64.(permutedims(channelview(rgbimg), (2, 3, 1))),)
        )
        SImageND(N0f8.(clamp01nan.(h))), SImageND(N0f8.(clamp01nan.(e))), SImageND(N0f8.(clamp01nan.(d)))
    end
    return h, e, d
end

# ################### #
# DATALOADERS ------- -
# ################### #

abstract type AbstractDataLoader end
struct DataLoader <: AbstractDataLoader
    path::String
    n::Int
    files
    batch_size::Int # For multi threading
    function DataLoader(path, n, bs)
        files = readdir(path)
        return new(path, n, files, bs)
    end
end

abstract type RAMDataLoader <: AbstractDataLoader end
struct RamDataLoader <: RAMDataLoader
    xs
    ys
    indices
    batch_size::Int # How many to send to a single T
    n::Int # How many to sample
end
struct HashRamDataLoader <: RAMDataLoader
    xs
    ys
    hs
    indices
    batch_size::Int # How many to send to a single T
    n::Int # How many to sample
    function HashRamDataLoader(xs, ys, batch_size)
        n = length(xs)
        @assert n == length(ys)
        indices = collect(1:n)
        return new(
            xs,
            ys,
            hash.(xs),
            indices,
            batch_size,
            n
        )

    end
end

function Base.iterate(dl::HashRamDataLoader, state = 1)
    if state > dl.n
        return nothing
    end
    return ((dl.xs[state], dl.ys[state], dl.hs[state]), state + 1)
end

function Base.firstindex(dl::HashRamDataLoader)
    return firstindex(dl.xs)
end
function Base.getindex(dl::HashRamDataLoader, i::Int)
    return (dl.xs[i], dl.ys[i], dl.hs[i])
end

struct WRamDataLoader <: RAMDataLoader
    xs
    ys
    indices
    batch_size::Int # How many to send to a single T
    n::Int # How many to sample
    ws::Vector{Float64}
    function WRamDataLoader(xs, ys, indices, batch_size, n)
        ws = ones(Float64, length(xs))
        return new(xs, ys, indices, batch_size, n, ws)
    end
end

# Methods Dataloaders
Base.length(d::AbstractDataLoader) = d.n

function Base.getindex(d::RAMDataLoader, I::UnitRange{Int})
    labels = d.ys #map(x -> x[1], d.ys)
    classes = unique(labels)
    n_classes = length(classes)
    how_many = length(I)
    how_many_per_class = trunc(Int, how_many / n_classes)
    @assert how_many % n_classes == 0 "$how_many $n_classes"
    # Collect sampled indices for each class
    sampled_indices = Int[]
    for c in classes
        # Find indices for class c
        c_idx = d.indices[labels .== c]
        # Sample with replacement if needed
        sampled = sample(c_idx, how_many_per_class; replace = true)
        append!(sampled_indices, sampled)
    end
    shuffle!(sampled_indices)
    xs = d.xs[sampled_indices]
    ys = d.ys[sampled_indices]
    return collect(zip(xs, ys)), sampled_indices
end

function Main.sample(d::RAMDataLoader, how_much::Int; balanced::Bool = true)
    balanced = d.ys isa Vector{Float64} ? false : balanced
    if balanced
        return d[1:how_much] # calls getindex which is balanced
    end
    idx = sample(d.indices, how_much)
    xs = d.xs[idx]
    ys = d.ys[idx]
    return collect(zip(xs, ys)), idx
end

function Main.sample(d::HashRamDataLoader, how_much::Int)
    idx = sample(d.indices, how_much)
    xs = d.xs[idx]
    ys = d.ys[idx]
    hs = d.hs[idx]
    return collect(zip(xs, ys, hs)), idx
end

function Base.getindex(d::WRamDataLoader, I::UnitRange{Int})
    how_many = length(I)
    which_samples = sample(d.indices, Weights(d.ws), how_many, replace = false)
    try
        println("Dataloader Histogram")
        histogram(d.ws) |> println
    catch
        println("Could not print histogram")
    end
    xs = d.xs[which_samples]
    ys = d.ys[which_samples]
    return which_samples, collect(zip(xs, ys))
end

function make_dataloader(x, y, nsamples, nt)
    return RamDataLoader(
        x, y,
        collect(1:length(y)),
        ceil(Int, nsamples / nt), nsamples
    )
end

# HUNCRC
# https://www.nature.com/articles/s41597-022-01450-y

const HUNCRC_LABELS::Vector{String} = [
    "highgrade_dysplasia", "adenocarcinoma", "suspicious_for_invasion",
    "lymphovascular_invasion", "inflammation", "resection_edge",
    "tumor_necrosis", "artifact", "normal", "lowgrade_dysplasia",
]
const HUNCRC_BENIGN::Vector{String} = ["lowgrade_dysplasia", "inflammation", "resection_edge", "artifact", "normal"]
const HUNCRC_MALIGN::Vector{String} = ["highgrade_dysplasia", "adenocarcinoma", "suspicious_for_invasion", "tumor_necrosis", "lymphovascular_invasion"]
@assert Set(union(HUNCRC_BENIGN, HUNCRC_MALIGN)) == Set(HUNCRC_LABELS)
@assert isempty(intersect(HUNCRC_MALIGN, HUNCRC_BENIGN))

# ################################# #
# LOAD DATASET TO MEM UTILS ------- -
# ################################# #

function _display_class_distribution(y_data::Vector{Int}, split_name::String)
    tmp = countmap(y_data)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = split_name) |> println
    return println("$split_name Labels: $tmp")
end

function _process_image_channels(img, include_gray::Bool, include_hsv::Bool, include_hed::Bool; resize_to::Union{Tuple{Int, Int}, Nothing} = nothing)
    channels = Any[]

    if !isnothing(resize_to)
        img = imresize(img, resize_to)
    end

    # RGB Channels
    rgb_channels = collect(channelview(img))
    for i in 1:3
        push!(channels, SImageND(IntensityPixel{N0f8}.(rgb_channels[i, :, :])))
    end

    if include_gray
        # Grayscale Channel
        push!(channels, SImageND(IntensityPixel{N0f8}.(clamp01nan.(Gray.(img)))))
    end

    # HSV Channels
    if include_hsv
        hsv_img = HSV.(img)
        hsv_channels = collect(channelview(float(hsv_img)))
        # Hue channel needs to be normalized to [0, 1]
        push!(channels, SImageND(IntensityPixel{N0f8}.(clamp01nan.(hsv_channels[1, :, :] / 360.0))))
        push!(channels, SImageND(IntensityPixel{N0f8}.(hsv_channels[2, :, :])))
        push!(channels, SImageND(IntensityPixel{N0f8}.(hsv_channels[3, :, :])))
    end

    # HED Channels (optional)
    if include_hed
        heds = rgb2hed(img)
        push!(channels, heds[1]) # H channel
        push!(channels, heds[2]) # E channel
        push!(channels, heds[3]) # D channel
    end
    return channels
end


function load_diagset_dataset(
        dataset_base_path::String, # "datasets"
        zoom_level::String; # e.g., "5x", "10x", "20x", "40x"
        include_hed::Bool = false,
        stratify_p = (0.2, 0.2) # Not strictly needed if partitions exist, but good for consistency
    )
    @info "Stratify proportion will be omitted since this dataset already has splits"

    base_image_path = joinpath(dataset_base_path, "DiagSet-A", "blobs", "S", zoom_level)
    partitions_path = joinpath(dataset_base_path, "DiagSet-A", "partitions", "DiagSet-A.2")

    # Read partitions
    train_ids = readdlm(joinpath(partitions_path, "train.csv"), String)[2:end, 1] # remove col header
    val_ids = readdlm(joinpath(partitions_path, "validation.csv"), String)[2:end, 1] # remove col header
    test_ids = readdlm(joinpath(partitions_path, "test.csv"), String)[2:end, 1] # remove col header

    # Identify classes and map image IDs to labels
    image_id_to_label = Dict{String, String}()
    all_image_paths_with_labels = []

    for slide_id in readdir(base_image_path) # enter the slide
        slide_folder_path = joinpath(base_image_path, slide_id)
        for class_folder in readdir(slide_folder_path) # enter the slide + class
            class_folder_path = joinpath(slide_folder_path, class_folder)
            if isdir(class_folder_path)
                for image_file in readdir(class_folder_path) # enter the images
                    image_id = splitext(image_file)[1] # Get UUID part of filename
                    image_id_to_label[joinpath(class_folder_path, image_file)] = class_folder
                    push!(
                        all_image_paths_with_labels,
                        (joinpath(class_folder_path, image_file), class_folder, slide_id)
                    )
                end
            end
        end
    end

    # Create a mapping from label name (string) to integer (1-indexed)
    unique_labels = sort(unique(values(image_id_to_label)))
    label_to_int = Dict(label => i for (i, label) in enumerate(unique_labels))

    @show label_to_int

    # Helper function to load and process images for a given set of IDs
    function _load_split_data(ids::Vector{String})
        x_data = Any[]
        y_data = Vector{Int}[]
        for id in ids
            found_image = false
            for (path, label, image_id) in all_image_paths_with_labels
                if image_id == id
                    if id == "35B51168-1B35-47FE-9117-E7E5A3D1AE52"
                        if isdefined(Main, :Infiltrator)
                            Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
                        end
                    end
                    raw_imgs_batch = npzread(path)
                    num_images_in_batch = size(raw_imgs_batch, 1)
                    for i in 1:num_images_in_batch # Each npy file has multiple image from the blob region
                        # Each img is 256x256. Format is (N, H, W, C) C is RGB
                        single_raw_img = clamp01nan.(float(raw_imgs_batch[i, :, :, :]) ./ 255.0)
                        img_rgb = colorview(RGB, permutedims(single_raw_img, (3, 1, 2)))
                        processed_channels = _process_image_channels(img_rgb, include_hed; resize_to = (32, 32))
                        push!(x_data, processed_channels)
                        push!(y_data, label_to_int[label]) # All patches from this .npy have the same label
                    end
                    found_image = true
                end
            end
            if !found_image
                @warn "Image with ID $id not found in image directories for zoom level $zoom_level."
            end
        end
        return x_data, y_data
    end

    train_x, train_y = _load_split_data(train_ids)
    val_x, val_y = _load_split_data(val_ids)
    test_x, test_y = _load_split_data(test_ids)

    all_x = [train_x..., val_x..., test_x...]
    all_y = [train_y..., val_y..., test_y...]

    # Display class distribution
    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return train_x, train_y, val_x, val_y, test_x, test_y, all_x, all_y, Dict(
            "unique_labels" => unique_labels,
            "all_image_paths_with_labels" => all_image_paths_with_labels,
            "label_to_int" => label_to_int,
            "image_id_to_label" => image_id_to_label
        )
end

################# IMAGEWOOF   ####################
function load_imagewoof(
        data_path::String;
        resize_to::Union{Tuple{Int, Int}, Nothing} = (160, 160),
        split_ratio = 0.8
    )

    folder_mapping_to_breed = Dict(
        "n02096294" => "Australian terrier", "n02093754" => "Border terrier", "n02111889" => "Samoyed",
        "n02088364" => "Beagle", "n02086240" => "Shih-Tzu", "n02089973" => "English foxhound",
        "n02087394" => "Rhodesian ridgeback", "n02115641" => "Dingo",
        "n02099601" => "Golden retriever", "n02105641" => "Old English sheepdog"
    )
    folder_mapping_to_int = Dict(k => i for (i, k) in enumerate(keys(folder_mapping_to_breed)))

    @info "Loading ImageWoof from $data_path"
    train_dir = joinpath(data_path, "train")
    test_dir = joinpath(data_path, "val")

    folders_to_read = sort(readdir(train_dir))
    @assert folders_to_read == sort(readdir(test_dir))
    train_folders = Set{String}()
    test_folders = Set{String}()
    train_data = []
    test_data = []

    for (split_dir, storage) in ((train_dir, train_data), (test_dir, test_data))
        for folder in folders_to_read
            p = joinpath(split_dir, folder)
            imgs = readdir(p)
            breed = folder_mapping_to_breed[folder]
            breed_int = folder_mapping_to_int[folder]
            for img in imgs
                full_path_img = joinpath(p, img)
                loaded_img = load(full_path_img)
                if eltype(loaded_img) == Gray{N0f8} # img was 2D gray not RGB ...
                    tmp = zeros(eltype(loaded_img), size(loaded_img, 1), size(loaded_img, 2), 3) # repeat BW 3 times
                    tmp[:, :, 1] .= loaded_img
                    tmp[:, :, 2] .= loaded_img
                    tmp[:, :, 3] .= loaded_img
                    loaded_img = colorview(RGB, permutedims(tmp, (3, 1, 2)))
                end
                processed_xs = _process_image_channels(loaded_img, false, false, false; resize_to = resize_to)
                push!(storage, (img = processed_xs, filename = img, path = full_path_img, folder = folder, breed = breed, class = breed_int))
            end
        end
    end

    @info "Number of images in train : $(length(train_data))"
    @info "Number of images in test : $(length(test_data))"

    # Step 5: Load and process images for each split
    new_train_data, new_val_data = MLUtils.splitobs(Xoshiro(1), train_data; at = split_ratio, shuffle = false, stratified = map(x -> x.class, train_data))
    train_x, train_y = map(x -> x.img, new_train_data), map(x -> x.class, new_train_data)
    val_x, val_y = map(x -> x.img, new_val_data), map(x -> x.class, new_val_data)
    test_x, test_y = map(x -> x.img, test_data), map(x -> x.class, test_data)

    train_x, train_y = identity.(train_x), identity.(train_y)
    val_x, val_y = identity.(val_x), identity.(val_y)
    test_x, test_y = identity.(test_x), identity.(test_y)

    all_x_data = [train_x..., val_x..., test_x...]
    all_y_data = [train_y..., val_y..., test_y...]

    # Step 6: Display class distribution (optional)
    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    extras = (
        folder_mapping_to_breed = folder_mapping_to_breed,
        folder_mapping_to_int = folder_mapping_to_int,
        train_paths = map(x -> x.filename, new_train_data),
        val_paths = map(x -> x.filename, new_val_data),
        test_paths = map(x -> x.filename, test_data),
    )

    return train_x, train_y, val_x, val_y, test_x, test_y, all_x_data, all_y_data, extras
end

################# HF LF DLBCL ####################"

"""
Row Info per anonymized folder.
Folder => (label, split, patient)
"""
function _read_hf_lf_dlbcl_split_metadata(split_csv_path::String)
    split_df = CSV.read(split_csv_path, DataFrame)
    anonymized_to_label = Dict{String, NamedTuple}()
    for row in eachrow(split_df)
        anonymized_to_label[row.Anonymized] = (label = row.label, split = row.split, patient = row.patient)
    end
    return anonymized_to_label, split_df
end

function _collect_hf_lf_dlbcl_image_paths(base_dir::String, level_id::Int, anonymized_to_info::Dict)
    all_image_meta = NamedTuple[] # Stores (image_path, label, split, original_filename, anonymized_id_folder)

    for anonymized_id_folder in readdir(base_dir)
        if !isdir(joinpath(base_dir, anonymized_id_folder)) # there are the csv files which we don't walk
            continue
        end
        level_folder_path = joinpath(base_dir, anonymized_id_folder, "level_$(level_id)")
        if isdir(level_folder_path)
            for image_file in readdir(level_folder_path)
                if endswith(image_file, ".png")
                    image_path = joinpath(level_folder_path, image_file)
                    folder_info = anonymized_to_info[anonymized_id_folder]
                    label = folder_info.label
                    split_type = folder_info.split
                    push!(all_image_meta, (path = image_path, label = label, split = split_type, file_name = image_file, folder_name = anonymized_id_folder))
                end
            end
        else
            @warn "Level $(level_id) folder not found for $anonymized_id_folder. Skipping."
        end
    end
    return all_image_meta
end

function _load_and_process_image_split_hf_lf_dlbcl(
        image_meta_list::Vector,
        label_to_int::Dict,
        include_hed::Bool,
        resize_to::Union{Tuple{Int, Int}, Nothing}
    )
    x_data = Any[]
    y_data = Int[]
    for image_metadata in image_meta_list
        raw_img = PNGFiles.load(image_metadata.path)
        processed_channels = _process_image_channels(raw_img, include_hed; resize_to = resize_to)
        label_int = label_to_int[image_metadata.label]
        push!(x_data, processed_channels)
        push!(y_data, label_int)
    end
    return x_data, y_data
end

function load_hf_lf_dlbcl_dataset(
        dataset_base_path::String;
        folder_name::String,
        file_name::String,
        level_id::Int = 2, # Default to level_2
        include_hed::Bool = false,
        resize_to::Union{Tuple{Int, Int}, Nothing} = (32, 32)
    )
    @info "Loading HF_LF_DLBCL_for_Camilo dataset, level_$(level_id)"

    base_dir = joinpath(dataset_base_path, folder_name)
    split_csv_path = joinpath(base_dir, file_name)

    # Step 1: Read split metadata
    anonymized_to_info, split_df = _read_hf_lf_dlbcl_split_metadata(split_csv_path)

    # Step 2: Collect all image paths with their metadata
    all_image_meta = _collect_hf_lf_dlbcl_image_paths(base_dir, level_id, anonymized_to_info)

    # Step 3: Create a mapping from label name (string) to integer (1-indexed)
    unique_labels = sort(unique([m.label for m in all_image_meta]))
    @info "Unique Labels : $unique_labels"
    label_to_int = Dict(label => i for (i, label) in enumerate(unique_labels))
    @info "Labels in Int : $label_to_int"

    # Step 4: Filter image meta for each split
    train_image_meta = filter(m -> m.split == "train", all_image_meta)
    val_image_meta = filter(m -> m.split == "val", all_image_meta)
    test_image_meta = filter(m -> m.split == "test", all_image_meta)

    # Step 5: Load and process images for each split
    train_x, train_y = _load_and_process_image_split_hf_lf_dlbcl(train_image_meta, label_to_int, include_hed, resize_to)
    val_x, val_y = _load_and_process_image_split_hf_lf_dlbcl(val_image_meta, label_to_int, include_hed, resize_to)
    test_x, test_y = _load_and_process_image_split_hf_lf_dlbcl(test_image_meta, label_to_int, include_hed, resize_to)

    all_x_data = [train_x..., val_x..., test_x...]
    all_y_data = [train_y..., val_y..., test_y...]

    # Step 6: Display class distribution (optional)
    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    extras = Dict(
        "unique_labels" => unique_labels,
        "label_to_int" => label_to_int,
        "split_dataframe" => split_df,
        "train_metadata" => train_image_meta,
        "val_metadata" => val_image_meta,
        "test_metadata" => test_image_meta,
    )

    return train_x, train_y, val_x, val_y, test_x, test_y, all_x_data, all_y_data, extras
end

################# HF LF ####################

function _collect_hf_lf_image_paths(base_images_folder::String, anonymized_to_info::Dict)
    all_image_meta = NamedTuple[] # Stores (image_path, label, split, original_filename, anonymized_id_folder)

    if !isdir(base_images_folder)
        @warn "Base images folder $(base_images_folder) not found. Skipping."
        return all_image_meta
    end

    # Iterate through folders like "benign_001", "malign_001" within base_images_folder (e.g., "datasets/patches_camilo/HF")
    for patient_folder_name in readdir(base_images_folder)
        patient_folder_path = joinpath(base_images_folder, patient_folder_name)

        if !isdir(patient_folder_path)
            continue
        end

        # 'patient_folder_name' (e.g., "benign_001") is the key in anonymized_to_info from the CSV.
        if haskey(anonymized_to_info, patient_folder_name)
            folder_info = anonymized_to_info[patient_folder_name]
            label = folder_info.label
            split_type = folder_info.split

            for image_file in readdir(patient_folder_path)
                if endswith(image_file, ".png")
                    image_path = joinpath(patient_folder_path, image_file)
                    push!(all_image_meta, (path = image_path, label = label, split = split_type, file_name = image_file, folder_name = patient_folder_name))
                end
            end
        else
            @warn "Patient folder metadata for '$patient_folder_name' not found in annotation file. Skipping."
        end
    end
    return all_image_meta
end

function load_hf_lf_dataset(
        dataset_base_path::String; # e.g., "datasets/patches_camilo"
        file_name::String = "annotation_evostar.csv",
        include_hed::Bool = false,
        resize_to::Union{Tuple{Int, Int}, Nothing} = (32, 32)
    )
    @info "Loading HF_LF_dataset from $(dataset_base_path)"
    split_csv_path = joinpath(dataset_base_path, file_name)
    anonymized_to_info, split_df = _read_hf_lf_dlbcl_split_metadata(split_csv_path)
    all_image_meta_aggregated = NamedTuple[]

    # Iterate through top-level directories like "HF", "HF_1", "LF", "LF_1" within dataset_base_path
    for sub_folder_name in readdir(dataset_base_path)
        current_sub_folder_path = joinpath(dataset_base_path, sub_folder_name)
        if isdir(current_sub_folder_path) && sub_folder_name != splitext(file_name)[1]
            @info "Processing sub-folder: $(sub_folder_name)"
            current_image_meta = _collect_hf_lf_image_paths(current_sub_folder_path, anonymized_to_info)
            append!(all_image_meta_aggregated, current_image_meta)
        end
    end

    unique_labels = sort(unique([m.label for m in all_image_meta_aggregated]))
    @info "Unique Labels : $unique_labels"
    label_to_int = Dict(label => i for (i, label) in enumerate(unique_labels))
    @info "Labels in Int : $label_to_int"

    train_image_meta = filter(m -> m.split == "train", all_image_meta_aggregated)
    val_image_meta = filter(m -> m.split == "val", all_image_meta_aggregated)
    test_image_meta = filter(m -> m.split == "test", all_image_meta_aggregated)
    train_x, train_y = _load_and_process_image_split_hf_lf_dlbcl(train_image_meta, label_to_int, include_hed, resize_to)
    val_x, val_y = _load_and_process_image_split_hf_lf_dlbcl(val_image_meta, label_to_int, include_hed, resize_to)
    test_x, test_y = _load_and_process_image_split_hf_lf_dlbcl(test_image_meta, label_to_int, include_hed, resize_to)
    all_x_data = [train_x..., val_x..., test_x...]
    all_y_data = [train_y..., val_y..., test_y...]
    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    extras = Dict(
        "unique_labels" => unique_labels,
        "label_to_int" => label_to_int,
        "split_dataframe" => split_df,
        "train_metadata" => train_image_meta,
        "val_metadata" => val_image_meta,
        "test_metadata" => test_image_meta,
    )

    return train_x, train_y, val_x, val_y, test_x, test_y, all_x_data, all_y_data, extras
end

# ############################## #
# LOAD DATASET INTERFACE ------- -
# ############################## #

# # CIFAR
function load_cifar_dataset(
        train_pct::Float64 = 0.8
    )
    cifar_train = MLDatasets.CIFAR10(split = :train)
    cifar_train_y = cifar_train.targets .+ 1
    cifar_train = convert2image(CIFAR10, cifar_train.features)

    cifar_test = MLDatasets.CIFAR10(split = :test)
    cifar_test_y = cifar_test.targets .+ 1
    cifar_test = convert2image(CIFAR10, cifar_test.features)

    cifar_train = [
        [
                [
                    SImageND(
                        IntensityPixel{N0f8}.(
                            channelview(cifar_train[:, :, i])[c, :, :]
                        )
                    ) for c in 1:3
                ]...,
            ] for i in 1:size(cifar_train)[3]
    ]
    cifar_test = [
        [
                [
                    SImageND(
                        IntensityPixel{N0f8}.(
                            channelview(cifar_test[:, :, i])[c, :, :]
                        )
                    ) for c in 1:3
                ]...,
            ] for i in 1:size(cifar_test)[3]
    ]

    @assert length(cifar_train) == length(cifar_train_y)
    @assert length(cifar_test) == length(cifar_test_y)

    (cifar_train_x, cifar_train_y), (cifar_val_x, cifar_val_y) = MLUtils.splitobs((cifar_train, cifar_train_y); at = train_pct, shuffle = true, stratified = cifar_train_y)
    tmp = countmap(cifar_train_y)
    barplot(collect(keys(tmp)), collect(values(tmp)), title = "Train") |> println
    tmp = countmap(cifar_val_y)
    barplot(collect(keys(tmp)), collect(values(tmp)), title = "Validation") |> println
    tmp = countmap(cifar_test_y)
    barplot(collect(keys(tmp)), collect(values(tmp)), title = "Test") |> println

    allx, ally = [cifar_train_x..., cifar_val_x...], [cifar_train_y..., cifar_val_y...]
    extras = Dict()

    return (
        identity.(cifar_train_x), identity.(cifar_train_y),
        identity.(cifar_val_x), identity.(cifar_val_y),
        identity.(cifar_test), identity.(cifar_test_y),
        allx, ally, extras,
    )
end


function load_mnist_dataset(
        train_pct::Float64 = 0.8
    )
    mnist_train = MLDatasets.MNIST(split = :train)
    mnist_train_y = mnist_train.targets .+ 1
    mnist_train = convert2image(MNIST, mnist_train.features)

    mnist_test = MLDatasets.MNIST(split = :test)[:]
    mnist_test_y = mnist_test[2] .+ 1
    mnist_test = convert2image(MNIST, mnist_test.features)

    mnist_train = [[SImageND(IntensityPixel{N0f8}.(mnist_train[:, :, i]))] for i in 1:size(mnist_train)[3]]
    mnist_test = [[SImageND(IntensityPixel{N0f8}.(mnist_test[:, :, i]))] for i in 1:size(mnist_test)[3]]

    (mnist_train_x, mnist_train_y), (mnist_val_x, mnist_val_y) = MLUtils.splitobs((mnist_train, mnist_train_y); at = train_pct, shuffle = true, stratified = mnist_train_y)
    tmp = countmap(mnist_train_y)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Train") |> println
    tmp = countmap(mnist_val_y)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Validation") |> println
    tmp = countmap(mnist_test_y)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Test") |> println

    # mnist_train_y = [[i] for i in mnist_train_y]
    # mnist_val_y = [[i] for i in mnist_val_y]
    # mnist_test_y = [[i] for i in mnist_test_y]
    allx, ally = [mnist_train_x..., mnist_val_x...], [mnist_train_y..., mnist_val_y...]

    return (
        mnist_train_x, mnist_train_y, mnist_val_x, mnist_val_y, mnist_test, mnist_test_y, allx, ally,
    )

    # Add constants
    # constants = [0.0, -1.0, 0.5, 2.0, 10, 20.0, 30.0]
    # for container in (trainx, valx, testx)
    #     for obs in container
    #         push!(obs, constants...)
    #     end
    # end
end


function load_FashionMNIST_dataset(
        train_pct::Float64 = 0.8
    )
    train = MLDatasets.FashionMNIST(split = :train)
    train_y = train.targets .+ 1
    train = convert2image(FashionMNIST, train.features)
    @assert size(train)[end] == length(train_y)

    test = MLDatasets.FashionMNIST(split = :test)
    test_y = test.targets .+ 1
    test = convert2image(FashionMNIST, test.features)
    @assert size(test)[end] == length(test_y)

    train = [[SImageND(IntensityPixel{N0f8}.(train[:, :, i]))] for i in 1:size(train)[3]]
    test = [[SImageND(IntensityPixel{N0f8}.(test[:, :, i]))] for i in 1:size(test)[3]]

    @assert size(train[1][1]) == size(test[1][1])

    (train_x, train_y), (val_x, val_y) = MLUtils.splitobs((train, train_y); at = train_pct, shuffle = true, stratified = train_y)
    tmp = countmap(train_y)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Train") |> println
    tmp = countmap(val_y)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Validation") |> println
    tmp = countmap(test_y)
    UnicodePlots.barplot(collect(keys(tmp)), collect(values(tmp)), title = "Test") |> println

    allx, ally = [train_x..., val_x...], [train_y..., val_y...]

    extras = Dict()

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test), identity.(test_y), allx, ally, extras,
    )

    # Add constants
    # constants = [0.0, -1.0, 0.5, 2.0, 10, 20.0, 30.0]
    # for container in (trainx, valx, testx)
    #     for obs in container
    #         push!(obs, constants...)
    #     end
    # end
end

# LOAD PCAM ----
function load_pcam_dataset(
        dataset_base_path::String,
        hed::Bool = false,
    )
    function read_pcam_file(file_path, hed::Bool = false)
        y = parse(Int, split(file_path, "_")[end][1]) # get label
        x = PNGFiles.load(file_path) # load 3d img
        # xs = [i[32:(32 * 2 - 1), 32:(32 * 2 - 1)] for i in xs] # https://github.com/basveeling/pcam # crop
        processed_xs = _process_image_channels(x, hed)
        return processed_xs, y
    end
    function read_subset_pcam(dir::String, nt::Int)
        files = readdir(dir, sort = true)
        N = length(files)
        IMG_X = Vector(undef, N)
        IMG_Y = Vector{Int}(undef, N)
        range_files = 1:N
        partition_size = ceil(Int, N / nt)
        @info "Reading Subset with $N files. $partition_size per Thread."
        @sync for idx_subset in Iterators.partition(range_files, partition_size)
            part = collect(idx_subset)
            subset_files = @view files[part]
            vw_x = @view IMG_X[part]
            vw_y = @view IMG_Y[part]
            t = Threads.@spawn begin
                @info "Reading $(length(part)) files at $(threadid())"
                for (idx, real_idx) in enumerate(part)
                    file_path = joinpath(dir, subset_files[idx])
                    x, y = read_pcam_file(file_path, false)
                    vw_x[idx] = x
                    vw_y[idx] = y + 1
                end

            end
        end
        @info "Loading data from $dir DONE"
        return IMG_X, IMG_Y, files
    end
    nt = nthreads()
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_subset_pcam(train_dir, nt)
    val_x, val_y, val_files = read_subset_pcam(val_dir, nt)
    test_x, test_y, test_files = read_subset_pcam(test_dir, nt)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end


###################################################################
# MED MNIST #######################################################
###################################################################

function read_rgb_img(gray, hsv, hed, file_path)
    y = parse(Int, split(split(file_path, "_label")[2], ".")[1]) # get label First split xxxxx _label 10.png | Second split [10] . png
    x = PNGFiles.load(file_path) # load 3d img
    processed_xs = _process_image_channels(x, gray, hsv, hed)
    return processed_xs, y
end
function read_gray_img(file_path)
    y = parse(Int, split(split(file_path, "_label")[2], ".")[1]) # get label First split xxxxx _label 10.png | Second split [10] . png
    x = PNGFiles.load(file_path) # load 3d img
    return [SImageND(IntensityPixel{N0f8}.(x))], y
end

function read_split_folder(dir::String, nt::Int, img_reader)
    files = readdir(dir, sort = true)
    N = length(files)
    IMG_X = Vector(undef, N)
    IMG_Y = Vector{Int}(undef, N)
    range_files = 1:N
    partition_size = ceil(Int, N / nt)
    @info "Reading Subset with $N files. $partition_size per Thread."
    @sync for idx_subset in Iterators.partition(range_files, partition_size)
        part = collect(idx_subset)
        subset_files = @view files[part]
        vw_x = @view IMG_X[part]
        vw_y = @view IMG_Y[part]
        t = Threads.@spawn begin
            @info "Reading $(length(part)) files at $(threadid())"
            for (idx, real_idx) in enumerate(part)
                file_path = joinpath(dir, subset_files[idx])
                x, y = img_reader(file_path)
                vw_x[idx] = x
                vw_y[idx] = y + 1
            end

        end
    end
    @info "Loading data from $dir DONE"
    return IMG_X, IMG_Y, files
end

# PATHMNIST
function load_PathMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/PathMNIST",
        gray::Bool = false,
        hsv::Bool = false,
        hed::Bool = false,
    )
    nt = nthreads()
    img_reader = read_rgb_img $ (gray, hsv, hed)
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

# ChestMNIST
function load_ChestMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/ChestMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

# DermaMNIST
function load_DermaMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/DermaMNIST",
        gray::Bool = false, hsv::Bool = false, hed::Bool = false
    )
    nt = nthreads()
    img_reader = read_rgb_img $ (gray, hsv, hed)
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

# DermaMNIST
function load_BreastMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/BreastMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

# BloodMNIST
function load_BloodMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/BloodMNIST",
        gray::Bool = false, hsv::Bool = false, hed::Bool = false
    )
    nt = nthreads()
    img_reader = read_rgb_img $ (gray, hsv, hed)
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

#TissueMNIST
function load_TissueMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/TissueMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

# OCTMNIST
function load_OCTMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/OCTMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

# PneumoniaMNIST
function load_PneumoniaMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/PneumoniaMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

# RetinaMNIST
function load_RetinaMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/RetinaMNIST",
        gray::Bool = false, hsv::Bool = false, hed::Bool = false
    )
    nt = nthreads()
    img_reader = read_rgb_img $ (gray, hsv, hed)
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end


function load_OrganAMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/OrganAMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

function load_OrganCMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/OrganCMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end

function load_OrganSMNIST(
        dataset_base_path::String = "./datasets/medmnist/2D/OrganSMNIST",
    )
    nt = nthreads()
    img_reader = read_gray_img
    train_dir = joinpath(dataset_base_path, "train")
    val_dir = joinpath(dataset_base_path, "val")
    test_dir = joinpath(dataset_base_path, "test")

    train_x, train_y, train_files = read_split_folder(train_dir, nt, img_reader)
    val_x, val_y, val_files = read_split_folder(val_dir, nt, img_reader)
    test_x, test_y, test_files = read_split_folder(test_dir, nt, img_reader)

    all_x = [train_x..., val_x...]
    all_y = [train_y..., val_y...]

    _display_class_distribution(train_y, "Train")
    _display_class_distribution(val_y, "Validation")
    _display_class_distribution(test_y, "Test")

    return (
        identity.(train_x), identity.(train_y), identity.(val_x), identity.(val_y), identity.(test_x), identity.(test_y), all_x, all_y, Dict(
            :train_files => train_files,
            :val_files => val_files,
            :test_files => test_files
        ),
    )
end


#     range_files = 1:N
#     partition_size = ceil(Int, N / nt)
#     @info "Reading $N files from $dir. Each Thread with $partition_size files"
#     for idx_subset in Iterators.partition(range_files, partition_size)
#         part = collect(idx_subset)
#         subset_files = @view images_dir[part]
#         vw_x = @view IMG_X[part]
#         vw_y = @view IMG_Y[part]
#         @dspawn begin
#             @R subset_files
#             @W vw_x
#             @W vw_y
#             tid = Threads.threadid()
#             @debug "Thread $tid started reading $(length(part)) files"
#             read_files(subset_files, dir, vw_x, vw_y, false)
#             @debug "Thread $tid ended reading $(length(part)) files"
#         end label = "$(part[begin]) : $(part[end])"
#     end
#     final_task = @dspawn @R(@view IMG_X[:]) label = "READ X"
#     other_final = @dspawn @R(@view IMG_Y[:]) label = "READ Y"
#     fetch(final_task)
#     fetch(other_final)
#     @info "Loading data from $dir DONE"
#     return (IMG_X, IMG_Y)
# end


# function read_files(subset_files, dir, vw_x, vw_y, hed::Bool = false)
#     ys = [parse(Int, split(i, "_")[end][1]) for i in subset_files] # get label
#     xs = [PNGFiles.load(joinpath(dir, i)) for i in subset_files] # load 3d img
#     xs = [i[32:(32 * 2 - 1), 32:(32 * 2 - 1)] for i in xs] # https://github.com/basveeling/pcam # crop
#     # imgs are still in rgb so make it hed
#     if hed
#         heds = rgb2hed.(xs)
#     end
#     # HSV ?
#     hsvs = [HSV.(x) for x in xs]
#     hsvs = [collect(channelview(float(x))) for x in hsvs]
#     hsvs = [[x[1, :, :] / 360.0, x[2, :, :], x[3, :, :]] for x in hsvs]
#     HSVS = []
#     for (i, hsv) in enumerate(hsvs)
#         push!(HSVS, [N0f8.(x) for x in hsv])
#     end
#     E = length(hsvs)
#     HSVS = [[SImageND(IntensityPixel{N0f8}.(HSVS[i][j])) for j in 1:3] for i in 1:E]
#     #
#     Gray_imgs = [identity.(convert.(N0f8, Gray.(three_D_img))) for three_D_img in xs]
#     xs = [collect(channelview(x)) for x in xs] # to [2D, 2D, 2D]
#     E = length(xs)
#     xs = [[SImageND(IntensityPixel{N0f8}.(xs[i][j, :, :])) for j in 1:3] for i in 1:E]
#     for (i, l) in enumerate(xs)
#         push!(l, SImageND(IntensityPixel{N0f8}.(Gray_imgs[i])))
#     end
#     if hed
#         Xs = [[obs..., hsv..., hed...] for (obs, hsv, hed) in zip(xs, HSVS, heds)] #, 0.1, -0.1, 0.5, -0.5, -1.0, -2.0, 2.0
#     else
#         Xs = [[obs..., hsv...] for (obs, hsv) in zip(xs, HSVS)]
#     end
#     Ys = [[l + 1] for l in ys]
#     vw_x[:] = Xs
#     return vw_y[:] = Ys
# end

# --- Augment Data legacy --- #

# function get_augmented_data(d::RamDataLoader, I::UnitRange{Int}; augmenter_inplace = basic_augment!)
#     samples = deepcopy(Base.getindex(d, I))
#     for (x, y) in samples
#         augmenter_inplace(x)
#     end
#     return samples
# end

# function process_simage(simg::SImageND{S, T}, f::Function, args...) where {S, T}
#     new_img = f(reinterpret(simg.img), args...)
#     return SImageND(T.(new_img), S)
# end
# function flip(img::AbstractArray)
#     return reverse(img, dims = 2)
# end
# function add_noise(img, noise; σ = 0.01)
#     return clamp01.(img .+ σ .* noise)
# end
# function adjust_brightness(img, factor)
#     return clamp01.(img .* factor)
# end
# function zoom_crop(img, zoom_factor)
#     h, w = size(img, 1), size(img, 2)

#     # Compute new dimensions after zoom-in
#     new_h = round(Int, h / zoom_factor)
#     new_w = round(Int, w / zoom_factor)

#     # Compute top-left corner to crop center
#     top = div(h - new_h, 2)
#     left = div(w - new_w, 2)

#     # Crop to center
#     cropped = img[(top + 1):(top + new_h), (left + 1):(left + new_w)]

#     # Resize back to original size
#     return imresize(cropped, (h, w))
# end

# function basic_augment!(imgs::Vector)
#     if rand() < 0.3 # reverse
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, flip)
#             end
#         end
#     end
#     if rand() < 0.3 # add noise
#         noise = randn(Float32, size(imgs[1])) # TODO ?
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, add_noise, noise)
#             end
#         end
#     end
#     if rand() < 0.3 # bright/dark
#         factor = rand(0.8:0.1:1.2)
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, adjust_brightness, factor)
#             end
#         end
#     end
#     return if rand() < 0.3 # add noise
#         zoom_factor = rand(1.0:0.1:1.2)
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, zoom_crop, zoom_factor)
#             end
#         end
#     end
# end

# function Base.getindex(d::RamDataLoader, I::UnitRange{Int})
#     how_many = length(I)
#     which_samples = sample(d.indices, how_many, replace=false)
#     xs = d.xs[which_samples]
#     ys = d.ys[which_samples]
#     collect(zip(xs, ys))
# end
#
# --- Augment Data legacy --- #

# function get_augmented_data(d::RamDataLoader, I::UnitRange{Int}; augmenter_inplace = basic_augment!)
#     samples = deepcopy(Base.getindex(d, I))
#     for (x, y) in samples
#         augmenter_inplace(x)
#     end
#     return samples
# end

# function process_simage(simg::SImageND{S, T}, f::Function, args...) where {S, T}
#     new_img = f(reinterpret(simg.img), args...)
#     return SImageND(T.(new_img), S)
# end
# function flip(img::AbstractArray)
#     return reverse(img, dims = 2)
# end
# function add_noise(img, noise; σ = 0.01)
#     return clamp01.(img .+ σ .* noise)
# end
# function adjust_brightness(img, factor)
#     return clamp01.(img .* factor)
# end
# function zoom_crop(img, zoom_factor)
#     h, w = size(img, 1), size(img, 2)

#     # Compute new dimensions after zoom-in
#     new_h = round(Int, h / zoom_factor)
#     new_w = round(Int, w / zoom_factor)

#     # Compute top-left corner to crop center
#     top = div(h - new_h, 2)
#     left = div(w - new_w, 2)

#     # Crop to center
#     cropped = img[(top + 1):(top + new_h), (left + 1):(left + new_w)]

#     # Resize back to original size
#     return imresize(cropped, (h, w))
# end

# function basic_augment!(imgs::Vector)
#     if rand() < 0.3 # reverse
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, flip)
#             end
#         end
#     end
#     if rand() < 0.3 # add noise
#         noise = randn(Float32, size(imgs[1])) # TODO ?
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, add_noise, noise)
#             end
#         end
#     end
#     if rand() < 0.3 # bright/dark
#         factor = rand(0.8:0.1:1.2)
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, adjust_brightness, factor)
#             end
#         end
#     end
#     return if rand() < 0.3 # add noise
#         zoom_factor = rand(1.0:0.1:1.2)
#         for (i, x) in enumerate(imgs)
#             if x isa SImageND
#                 imgs[i] = process_simage(x, zoom_crop, zoom_factor)
#             end
#         end
#     end
# end

# function Base.getindex(d::RamDataLoader, I::UnitRange{Int})
#     how_many = length(I)
#     which_samples = sample(d.indices, how_many, replace=false)
#     xs = d.xs[which_samples]
#     ys = d.ys[which_samples]
#     collect(zip(xs, ys))
# end

# SAVE FROM HDF5 TO IMG  ---------
# images = h5open("datasets/PCAM/Orig/camelyonpatch_level_2_split_train_x.h5")
# labels = h5open("datasets/PCAM/Orig/camelyonpatch_level_2_split_train_y.h5")
# ys = labels["y"]
# xs = images["x"]
# x = [xs[:, :, :, i] for i in 1:size(xs)[end]]
# y = [ys[:, :, :, i][1] for i in 1:size(ys)[end]]
# n = 1
# for (img, class) in zip(x, y)
#     global n
#     colimg = colorview(RGB, reinterpret.(N0f8, img))
#     class_as_int = Int(class)
#     save("datasets/PCAM/Images/train/train_$(n)_$class_as_int.png", colimg)
#     n += 1
# end


# images = h5open("datasets/PCAM/Orig/camelyonpatch_level_2_split_valid_x.h5")
# labels = h5open("datasets/PCAM/Orig/camelyonpatch_level_2_split_valid_y.h5")
# ys = labels["y"]
# xs = images["x"]
# x = [xs[:, :, :, i] for i in 1:size(xs)[end]]
# y = [ys[:, :, :, i][1] for i in 1:size(ys)[end]]
# n = 1
# for (img, class) in zip(x, y)
#     global n
#     colimg = colorview(RGB, reinterpret.(N0f8, img))
#     class_as_int = Int(class)
#     save("datasets/PCAM/Images/val/train_$(n)_$class_as_int.png", colimg)
#     n += 1
# end

# images = h5open("datasets/PCAM/Orig/camelyonpatch_level_2_split_test_x.h5")
# labels = h5open("datasets/PCAM/Orig/camelyonpatch_level_2_split_test_y.h5")
# ys = labels["y"]
# xs = images["x"]
# x = [xs[:, :, :, i] for i in 1:size(xs)[end]]
# y = [ys[:, :, :, i][1] for i in 1:size(ys)[end]]
# n = 1
# for (img, class) in zip(x, y)
#     global n
#     colimg = colorview(RGB, reinterpret.(N0f8, img))
#     class_as_int = Int(class)
#     save("datasets/PCAM/Images/test/train_$(n)_$class_as_int.png", colimg)
#     n += 1
# end
