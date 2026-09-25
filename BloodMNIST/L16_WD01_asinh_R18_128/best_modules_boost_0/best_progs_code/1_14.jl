# module_key=1_14
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_14/L16_WD01_asinh_R18_128__2/checkpoint_0.pickle
# best_subproblem_loss=0.2999846542782155

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(orientation_energy_135, x1; return_type=Float64)
    tmp2 = safe_call(gaussian9_image2D, x3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(watershed_image2D, tmp2; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp4 = safe_call(glcm_ASM_maximum, tmp3; return_type=Float64)
    tmp5 = safe_call(add_img2D, x2, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(glcm_contrast_minimum, tmp5; return_type=Float64)
    tmp7 = safe_call(reduce_length, x3; return_type=Float64)
    tmp8 = safe_call(glcm_correlation_sum, x1, tmp7; return_type=Float64)
    tmp9 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_ASM_std, tmp9, tmp8; return_type=Float64)
    tmp11 = safe_call(glcm_dissimilarity_minimum, x3; return_type=Float64)
    tmp12 = safe_call(log10_, tmp11; return_type=Float64)
    tmp13 = safe_call(if_else_multiplexer, tmp12, tmp10, tmp6; return_type=Float64)
    tmp14 = safe_call(moffat5_image2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(orientation_coherence, tmp14; return_type=Float64)
    tmp16 = safe_call(erosion_2D, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(dilation_2D, tmp16, tmp11; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(experimental_invert_2D, tmp17; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp19 = safe_call(moffat25_image2D, tmp18, tmp15, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp21 = safe_call(glcm_correlation_sum, tmp20; return_type=Float64)
    tmp22 = safe_call(minpool_cross_blocks, tmp17, tmp21; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp23 = safe_call(watershed_image2D, tmp22, tmp19, tmp13; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp24 = safe_call(glcm_correlation_std, tmp23, tmp4, tmp1; return_type=Float64)
    tmp25 = safe_call(identity_float, tmp24; return_type=Float64)
    out1 = tmp25
    return (out1,)
end
