"""Load train/val/test arrays from disk for distillation runs."""
function load_distill_split_arrays(parsed_args)
    split_data = load_distill_image_splits(parsed_args)
    return split_data.trainx, split_data.trainy, split_data.valx, split_data.valy, split_data.testx, split_data.testy
end

"""Infer basic dataset metadata from training samples and labels."""
function infer_distill_dataset_metadata(trainx, trainy)
    classes = sort(unique(trainy))
    n_classes = length(classes)
    sample_img = trainx[1][1]
    n_ins = length(trainx[1])
    return classes, n_classes, sample_img, n_ins
end

"""Compute per-channel normalization statistics from training inputs."""
function compute_distill_channel_stats(trainx)
    stacked = map(i -> reduce((x, y) -> cat(x, y, dims = 3), map(x -> reinterpret(x.img), i)), trainx)
    return stack_and_stats(stacked)
end

"""Compute a numerically safe Pearson correlation, returning `NaN` for degenerate vectors."""
function safe_pearson_cor(xs, ys)
    x = Float64.(xs)
    y = Float64.(ys)
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    denom = sqrt(sum(abs2, x_centered) * sum(abs2, y_centered))
    return denom == 0.0 ? NaN : sum(x_centered .* y_centered) / denom
end

"""Print per-class correlation and mean-activation summaries for one latent neuron."""
function print_latent_class_correlations(ys, labels, classes, split_str, neuron_idx)
    y_values = Float64.(ys)
    class_correlations = Pair{Any, Float64}[]
    class_means = Pair{Any, Float64}[]
    for cls in classes
        mask = labels .== cls
        push!(class_correlations, cls => safe_pearson_cor(y_values, Float64.(mask)))
        push!(class_means, cls => any(mask) ? mean(y_values[mask]) : NaN)
    end
    sorted_correlations = sort(class_correlations; by = p -> (isnan(p.second) ? -Inf : abs(p.second)), rev = true)
    sorted_means = sort(class_means; by = p -> (isnan(p.second) ? -Inf : p.second), rev = true)
    println("Latent neuron class correlations | split=$split_str | neuron_idx=$neuron_idx")
    println("class_correlations = $(repr(class_correlations))")
    println("sorted_by_abs_correlation = $(repr(sorted_correlations))")
    println("Latent neuron class activation means | split=$split_str | neuron_idx=$neuron_idx")
    println("class_activation_means = $(repr(class_means))")
    return println("sorted_by_mean_activation = $(repr(sorted_means))")
end
