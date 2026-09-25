# module_key=1_13
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_13/L16_WD01_asinh_R18_128__3/checkpoint_0.pickle
# best_subproblem_loss=0.21473474690615135

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_correlation_minimum, tmp1; return_type=Float64)
    tmp3 = safe_call(gaussian9_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(glcm_glcm_entropy_maximum, tmp3; return_type=Float64)
    tmp5 = safe_call(erosion_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(sobelm_image2D, tmp5, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(argmaxcountpool, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(scharrx_image2D, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(subtract_img2D, tmp8, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(orientation_energy_135, tmp9; return_type=Float64)
    tmp11 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(glcm_energy_std, tmp11; return_type=Float64)
    tmp13 = safe_call(reduce_mean, x1; return_type=Float64)
    tmp14 = safe_call(region_median, x1, tmp13, tmp12; return_type=Float64)
    tmp15 = safe_call(if_else_multiplexer, tmp14, tmp10, tmp2; return_type=Float64)
    tmp16 = safe_call(identity_float, tmp15; return_type=Float64)
    out1 = tmp16
    return (out1,)
end
