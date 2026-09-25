# module_key=1_2
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_2/L16_WD01_asinh_R18_128__104/checkpoint_0.pickle
# best_subproblem_loss=0.13293512877967528

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(argmincountpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(opening_2D, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(reduce_histModeCount, tmp2; return_type=Float64)
    tmp4 = safe_call(glcm_glcm_var_ref_sum, x1; return_type=Float64)
    tmp5 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(haar_three_h, tmp5, tmp4, tmp3; return_type=Float64)
    tmp7 = safe_call(binarize_entropy2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(gaussian25_image2D, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(gaussian9_image2D, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(ando4x_image2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(glcm_glcm_entropy_mean, tmp10, tmp6; return_type=Float64)
    tmp12 = safe_call(mult_img2D, tmp9, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(region_sum_5p, tmp12, tmp4, tmp11; return_type=Float64)
    tmp14 = safe_call(region_mean_5p, x1, tmp13, tmp6; return_type=Float64)
    tmp15 = safe_call(reduce_propWhite, tmp2; return_type=Float64)
    tmp16 = safe_call(power_of, tmp15, tmp14; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
