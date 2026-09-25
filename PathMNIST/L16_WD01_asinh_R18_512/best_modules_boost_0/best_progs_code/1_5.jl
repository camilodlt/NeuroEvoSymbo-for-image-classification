# module_key=1_5
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_5/L16_WD01_asinh_R18_512__104/checkpoint_0.pickle
# best_subproblem_loss=0.32917144927032016

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_max_prob_std, x2; return_type=Float64)
    tmp2 = safe_call(morpholaplace_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(glcm_energy_std, tmp2, tmp1; return_type=Float64)
    tmp4 = safe_call(glcm_contrast_mean, x1; return_type=Float64)
    tmp5 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(glcm_ASM_minimum, tmp5; return_type=Float64)
    tmp7 = safe_call(minpool, x3, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(region_sum_5p, tmp7, tmp1, tmp4; return_type=Float64)
    tmp9 = safe_call(binarize_moments2D, x3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(subtract_img2D, x3, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(experimental_tosegment_image2D, tmp10; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(glcm_correlation_minimum, tmp11, tmp8, tmp3; return_type=Float64)
    tmp13 = safe_call(identity_float, tmp12; return_type=Float64)
    out1 = tmp13
    return (out1,)
end
