"""Apply distillation-specific argument normalizations expected by downstream code."""
function normalize_distill_args!(parsed_args::AbstractDict)
    parsed_args["l1_size"] = parsed_args["latent_dim"]
    haskey(parsed_args, "binarize") || (parsed_args["binarize"] = false)
    return parsed_args
end

"""Create and return the output folder where distillation artifacts are saved."""
function ensure_distill_output_folder(output_dir::AbstractString, trial_id::AbstractString, boost_round)
    isdir(output_dir) || mkdir(output_dir)
    trial_dir = joinpath(output_dir, trial_id)
    isdir(trial_dir) || mkdir(trial_dir)
    folder = joinpath(trial_dir, "nn_surrogates_boost_$boost_round")
    isdir(folder) || mkdir(folder)
    return folder
end

"""Load train/val/test image splits from JLD2 files and wrap them as `SImageND` samples."""
function load_distill_image_splits(parsed_args::AbstractDict)
    data_path = parsed_args["data_location"]
    data = load(data_path)["single_stored_object"]
    trainx, trainy = data.xs, data.ys
    trainx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in trainx]

    println(UnicodePlots.histogram(trainy, title = "TRAIN Labels histogram"))
    println("Labels : $(sort(unique(trainy)))")

    val_data_path = parsed_args["val_data_location"]
    test_data_path = parsed_args["test_data_location"]

    if val_data_path != ""
        data = load(val_data_path)["single_stored_object"]
        valx, valy = data.xs, data.ys
        valx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in valx]
        @assert length(valx) == length(valy)
        @assert sort(unique(trainy)) == sort(unique(valy))
    else
        valx, valy = nothing, nothing
    end

    if test_data_path != ""
        data = load(test_data_path)["single_stored_object"]
        testx, testy = data.xs, data.ys
        testx = [[SImageND(reinterpret.(IntensityPixel{N0f8}, i)) for i in x] for x in testx]
        @assert length(testx) == length(testy)
        @assert sort(unique(trainy)) == sort(unique(testy))
    else
        testx, testy = nothing, nothing
    end

    @assert length(trainx) == length(trainy)
    @info "Size train : $(length(trainx))"
    @info "Size val : $(length(valx))"
    @info "Size test : $(length(testx))"

    return (
        data_path = data_path,
        val_data_path = val_data_path,
        test_data_path = test_data_path,
        trainx = trainx,
        trainy = trainy,
        valx = valx,
        valy = valy,
        testx = testx,
        testy = testy,
    )
end

"""Stack image tensors and compute per-channel mean/std statistics for normalization."""
function stack_and_stats(imgs::Vector)
    first_img = imgs[1]
    nd = ndims(first_img)
    if nd == 2
        h, w = size(first_img)
        c = 1
        n = length(imgs)
        out = Array{N0f8}(undef, h, w, 1, n)
        @inbounds for i in 1:n
            out[:, :, 1, i] = imgs[i]
        end
    elseif nd == 3
        h, w, c = size(first_img)
        n = length(imgs)
        out = Array{N0f8}(undef, h, w, c, n)
        @inbounds for i in 1:n
            out[:, :, :, i] = imgs[i]
        end
    else
        error("Images must be 2D or 3D arrays.")
    end

    means = zeros(Float32, c)
    stds = zeros(Float32, c)
    @inbounds for ch in 1:c
        channel_data = @view out[:, :, ch, :]
        flat = Float32.(channel_data[:])
        means[ch] = mean(flat)
        stds[ch] = std(flat)
    end
    return means, stds
end

"""Build train/eval torchvision transforms from model and dataset preprocessing settings."""
function build_distill_transforms(sample_img, n_ins::Int; pretrained_model::String, use_imagenet_stats::Bool, resize_to::Int, means, stds)
    torchvision = pyimport("torchvision.transforms")

    train_base_transform = Any[]
    val_base_transform = Any[]
    img_size = size(sample_img, 1)
    effective_resize_to = resize_to

    if pretrained_model == "vit"
        @info "Since vit model, size must be 224"
        effective_resize_to = 224
    end
    if use_imagenet_stats
        @info "OLD MEANS : $means , $stds"
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        @info "NEW MEANS : $means , $stds"
    end

    if effective_resize_to != -1
        @info "Image will have to be resized for model : $effective_resize_to"
        push!(train_base_transform, torchvision.transforms.RandomResizedCrop(size = (effective_resize_to, effective_resize_to), scale = (0.8, 1.0), ratio = (0.9, 1.1)))
        push!(val_base_transform, torchvision.transforms.Resize(size = (effective_resize_to, effective_resize_to)))
        img_size = effective_resize_to
    else
        push!(train_base_transform, torchvision.transforms.RandomResizedCrop(size(sample_img, 1), scale = (0.8, 1.0), ratio = (0.9, 1.1)))
    end

    if n_ins == 1
        py_train_transforms = torchvision.Compose([
            train_base_transform...,
            torchvision.transforms.ColorJitter(brightness = 0.3, contrast = 0.3),
            torchvision.ToTensor(),
            torchvision.Normalize(mean = means, std = stds),
        ])
        py_val_transforms = torchvision.Compose([
            val_base_transform...,
            torchvision.ToTensor(),
            torchvision.Normalize(mean = means, std = stds),
        ])
    else
        py_train_transforms = torchvision.Compose([
            train_base_transform...,
            torchvision.transforms.RandomHorizontalFlip(p = 0.5),
            torchvision.transforms.RandomVerticalFlip(p = 0.1),
            torchvision.transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.1, hue = 0.02),
            torchvision.ToTensor(),
            torchvision.Normalize(mean = means, std = stds),
        ])
        py_val_transforms = torchvision.Compose([
            val_base_transform...,
            torchvision.ToTensor(),
            torchvision.Normalize(mean = means, std = stds),
        ])
    end

    println(py_train_transforms)
    println(py_val_transforms)

    return (
        means = means,
        stds = stds,
        img_size = img_size,
        resize_to = effective_resize_to,
        py_train_transforms = py_train_transforms,
        py_val_transforms = py_val_transforms,
    )
end

"""Create PyTorch datasets/dataloaders for distillation training-time loops."""
function build_distill_dataloaders(pylipext, trainx, trainy, valx, valy, testx, testy, py_train_transforms, py_val_transforms, bs_train::Int, bs_val::Int; workers::Bool)
    image_dataset_class = pylipext.get_ImageDataset()
    image_dataloader_class = pylipext.get_ImageDataLoader()

    image_arrays = build_distill_image_arrays(trainx, valx, testx)
    vec_of_3d_imgs_train = image_arrays.vec_of_3d_imgs_train
    vec_of_3d_imgs_val = image_arrays.vec_of_3d_imgs_val
    vec_of_3d_imgs_test = image_arrays.vec_of_3d_imgs_test

    train_dataset = image_dataset_class(vec_of_3d_imgs_train, trainy .- 1, py_train_transforms)
    val_dataset = image_dataset_class(vec_of_3d_imgs_val, valy .- 1, py_val_transforms)
    test_dataset = image_dataset_class(vec_of_3d_imgs_test, testy .- 1, py_val_transforms)

    test_dataloader = image_dataloader_class(test_dataset, batch_size = bs_val, shuffle = false, drop_last = false)
    if workers
        train_dataloader = image_dataloader_class(train_dataset, batch_size = bs_train, shuffle = true, drop_last = true, num_workers = 4, pin_memory = true, prefetch_factor = 2, persistent_workers = false)
        val_dataloader = image_dataloader_class(val_dataset, batch_size = bs_val, shuffle = true, drop_last = true, num_workers = 4, pin_memory = true, prefetch_factor = 2, persistent_workers = false)
    else
        train_dataloader = image_dataloader_class(train_dataset, batch_size = bs_train, shuffle = true, drop_last = true)
        val_dataloader = image_dataloader_class(val_dataset, batch_size = bs_val, shuffle = true, drop_last = true)
    end

    return (
        image_dataset_class = image_dataset_class,
        image_dataloader_class = image_dataloader_class,
        vec_of_3d_imgs_train = vec_of_3d_imgs_train,
        vec_of_3d_imgs_val = vec_of_3d_imgs_val,
        vec_of_3d_imgs_test = vec_of_3d_imgs_test,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        test_dataset = test_dataset,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        test_dataloader = test_dataloader,
    )
end

"""Convert MAGE image samples into UInt8 HWC arrays expected by Python datasets."""
function build_distill_image_arrays(trainx, valx, testx)
    vec_of_3d_imgs_train = map(i -> reduce((x, y) -> cat(x, y, dims = 3), map(x -> reinterpret.(UInt8, reinterpret(x.img)), i)), trainx)
    vec_of_3d_imgs_val = map(i -> reduce((x, y) -> cat(x, y, dims = 3), map(x -> reinterpret.(UInt8, reinterpret(x.img)), i)), valx)
    vec_of_3d_imgs_test = map(i -> reduce((x, y) -> cat(x, y, dims = 3), map(x -> reinterpret.(UInt8, reinterpret(x.img)), i)), testx)
    return (
        vec_of_3d_imgs_train = vec_of_3d_imgs_train,
        vec_of_3d_imgs_val = vec_of_3d_imgs_val,
        vec_of_3d_imgs_test = vec_of_3d_imgs_test,
    )
end

"""Create deterministic non-shuffled dataloaders used for capture/export phases."""
function build_eval_distill_dataloaders(pylipext, vec_of_3d_imgs_train, trainy, vec_of_3d_imgs_val, valy, vec_of_3d_imgs_test, testy, py_val_transforms, bs_train::Int, bs_val::Int)
    image_dataset_class = pylipext.get_ImageDataset()
    image_dataloader_class = pylipext.get_ImageDataLoader()

    train_dataset = image_dataset_class(vec_of_3d_imgs_train, trainy .- 1, py_val_transforms)
    val_dataset = image_dataset_class(vec_of_3d_imgs_val, valy .- 1, py_val_transforms)
    test_dataset = image_dataset_class(vec_of_3d_imgs_test, testy .- 1, py_val_transforms)

    train_dataloader = image_dataloader_class(
        train_dataset,
        batch_size = bs_train,
        shuffle = false,
        drop_last = false
    )
    val_dataloader = image_dataloader_class(
        val_dataset,
        batch_size = bs_val,
        shuffle = false,
        drop_last = false
    )
    test_dataloader = image_dataloader_class(
        test_dataset,
        batch_size = bs_val,
        shuffle = false,
        drop_last = false
    )

    return (
        image_dataset_class = image_dataset_class,
        image_dataloader_class = image_dataloader_class,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        test_dataset = test_dataset,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        test_dataloader = test_dataloader,
    )
end
