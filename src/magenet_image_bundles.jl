fallback_intensity() = SImageND(IntensityPixel{N0f8}.(ones(N0f8, size(sample_img))))
fallback_binary() = SImageND(BinaryPixel{Bool}.(ones(Bool, size(sample_img))))
fallback_segment() = SImageND(SegmentPixel{Int}.(ones(Int, size(sample_img))))

function _bundle_flag_enabled(parsed_args, key::String)
    return !isnothing(parsed_args) && haskey(parsed_args, key) && parsed_args[key]
end

function add_requested_image_extension_bundles!(
    image_intensity,
    image_binary,
    image_segment;
    parsed_args = nothing,
)
    if _bundle_flag_enabled(parsed_args, "use_new_intensityimg_extensions")
        bundles = UTCGP.get_extension_intensityimg()
        push!(image_intensity, bundles...)
        @info "Added new image extension bundles" bundle_type = "intensityimg" count = length(bundles)
    end
    if _bundle_flag_enabled(parsed_args, "use_new_binaryimg_extensions")
        bundles = UTCGP.get_extension_binaryimg()
        push!(image_binary, bundles...)
        @info "Added new image extension bundles" bundle_type = "binaryimg" count = length(bundles)
    end
    if _bundle_flag_enabled(parsed_args, "use_new_segmentimg_extensions")
        bundles = UTCGP.get_extension_segmentimg()
        push!(image_segment, bundles...)
        @info "Added new image extension bundles" bundle_type = "segmentimg" count = length(bundles)
    end
    return nothing
end

function add_requested_float_extension_bundles!(float_bundles; parsed_args = nothing)
    if _bundle_flag_enabled(parsed_args, "use_new_number_extensions")
        bundles = UTCGP.get_extension_nb()
        push!(float_bundles, bundles...)
        @info "Added new float extension bundles" bundle_type = "number" count = length(bundles)
    end
    if _bundle_flag_enabled(parsed_args, "use_imagegraph_bundle")
        push!(float_bundles, bundle_float_imagegraph)
        @info "Added new float extension bundles" bundle_type = "imagegraph" count = 1
    end
    return float_bundles
end

image_intensity = UTCGP.get_image2Dintensity_factory_bundles();
image_binary = UTCGP.get_image2Dbinary_factory_bundles();
image_segment = UTCGP.get_image2Dsegment_factory_bundles();
add_requested_image_extension_bundles!(
    image_intensity,
    image_binary,
    image_segment;
    parsed_args = isdefined(Main, :Parsed_args) ? Main.Parsed_args : nothing,
)
Type2Dimg_intensity = typeof(fallback_intensity())
Type2Dimg_binary = typeof(fallback_binary())
Type2Dimg_segment = typeof(fallback_segment())
@show Type2Dimg_intensity Type2Dimg_binary Type2Dimg_segment

# push!(image2D, UTCGP.experimental_bundle_image2D_mask_factory) # TODO

function set_bundle_casters!(bundles, caster)
    for factory_bundle in bundles
        for (i, wrapper) in enumerate(factory_bundle)
            wrapper.caster = caster
        end
    end
    return
end

for (factories, fallback, typeimg) in zip(
        [image_intensity, image_binary, image_segment],
        [fallback_intensity, fallback_binary, fallback_segment],
        [Type2Dimg_intensity, Type2Dimg_binary, Type2Dimg_segment],
    )
    for factory_bundle in factories
        for (i, wrapper) in enumerate(factory_bundle)
            fn = wrapper.fn(typeimg) # specialize
            wrapper.fallback = fallback
            factory_bundle.functions[i] =
                UTCGP.FunctionWrapper(
                    fn,
                    wrapper.name,
                    wrapper.caster,
                    wrapper.fallback;
                    description = wrapper.description,
                )
        end
    end
end
