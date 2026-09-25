# module_key=1_10
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_10/L16_WD01_asinh_R18_128__102/checkpoint_0.pickle
# best_subproblem_loss=0.16448928154335496

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(minpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_minimum, tmp1; return_type=Float64)
    tmp3 = safe_call(morpholaplace_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(morpholaplace_2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(glcm_max_prob_std, tmp4, tmp2; return_type=Float64)
    tmp6 = safe_call(identity_float, tmp5; return_type=Float64)
    tmp7 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp8 = safe_call(glcm_correlation_maximum, tmp7; return_type=Float64)
    tmp9 = safe_call(erosion_2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(erosion_2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(glcm_contrast_std, tmp10, tmp8, tmp2; return_type=Float64)
    tmp12 = safe_call(experimental_invert_2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(glcm_energy_mean, tmp12; return_type=Float64)
    tmp14 = safe_call(glcm_max_prob_sum, x1, tmp13, tmp11; return_type=Float64)
    tmp15 = safe_call(safe_div, tmp14, tmp6; return_type=Float64)
    tmp16 = safe_call(identity_float, tmp15; return_type=Float64)
    out1 = tmp16
    return (out1,)
end
