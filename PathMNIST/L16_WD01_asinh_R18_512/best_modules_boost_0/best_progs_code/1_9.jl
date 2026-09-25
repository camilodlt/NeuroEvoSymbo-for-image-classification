# module_key=1_9
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_9/L16_WD01_asinh_R18_512__104/checkpoint_0.pickle
# best_subproblem_loss=0.3355112297680769

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_correlation_mean, x2; return_type=Float64)
    tmp2 = safe_call(pi_, ; return_type=Float64)
    tmp3 = safe_call(reduce_histModeCount, x2; return_type=Float64)
    tmp4 = safe_call(erosion_2D, x2, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(argmaxcountpool, tmp4, tmp3, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(erosion_2D, tmp5, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_correlation_maximum, tmp6; return_type=Float64)
    tmp8 = safe_call(experimental_not, tmp1; return_type=Float64)
    tmp9 = safe_call(glcm_glcm_entropy_minimum, x1; return_type=Float64)
    tmp10 = safe_call(subtract_img2D, x1, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(region_sum, tmp10, tmp2, tmp9; return_type=Float64)
    tmp12 = safe_call(subtract_img2D, x1, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(exp_image2D, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(loginv_image2D, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(glcm_dissimilarity_std, tmp14, tmp11, tmp8; return_type=Float64)
    tmp16 = safe_call(power_of, tmp15, tmp7; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
