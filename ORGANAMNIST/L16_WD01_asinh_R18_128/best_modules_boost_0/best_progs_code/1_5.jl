# module_key=1_5
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_5/L16_WD01_asinh_R18_128__101/checkpoint_0.pickle
# best_subproblem_loss=0.28603614620582896

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(exp_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(laplacian3_image2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(minpool, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(prewittm_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(glcm_correlation_maximum, tmp4; return_type=Float64)
    tmp6 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_ASM_minimum, tmp6; return_type=Float64)
    tmp8 = safe_call(region_min, tmp1, tmp7, tmp7; return_type=Float64)
    tmp9 = safe_call(glcm_IDM_minimum, x1; return_type=Float64)
    tmp10 = safe_call(dilation_2D, x1, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(opening_2D, tmp10, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(reduce_mean, tmp11; return_type=Float64)
    tmp13 = safe_call(dilation_2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(region_median_20p, tmp13, tmp12, tmp8; return_type=Float64)
    tmp15 = safe_call(safe_div, tmp14, tmp5; return_type=Float64)
    tmp16 = safe_call(identity_float, tmp15; return_type=Float64)
    out1 = tmp16
    return (out1,)
end
