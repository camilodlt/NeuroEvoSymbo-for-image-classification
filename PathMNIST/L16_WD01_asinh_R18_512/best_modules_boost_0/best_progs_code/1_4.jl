# module_key=1_4
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_4/L16_WD01_asinh_R18_512__103/checkpoint_0.pickle
# best_subproblem_loss=0.31261049959670617

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_glcm_entropy_maximum, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_glcm_mean_ref_sum, x3; return_type=Float64)
    tmp4 = safe_call(binarize_adaptive2D, x1, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(medianpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(max_img2D, tmp5, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_max_prob_minimum, tmp6, tmp2; return_type=Float64)
    tmp8 = safe_call(erosion_2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(reduce_median, tmp8; return_type=Float64)
    tmp10 = safe_call(glcm_contrast_maximum, x1; return_type=Float64)
    tmp11 = safe_call(tophat_2D, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(glcm_glcm_mean_ref_minimum, tmp11, tmp10, tmp9; return_type=Float64)
    tmp13 = safe_call(number_mult, tmp9, tmp12; return_type=Float64)
    tmp14 = safe_call(number_div, tmp13, tmp7; return_type=Float64)
    tmp15 = safe_call(identity_float, tmp14; return_type=Float64)
    out1 = tmp15
    return (out1,)
end
