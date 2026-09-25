# module_key=1_11
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_11/L16_WD01_asinh_R18_512__103/checkpoint_0.pickle
# best_subproblem_loss=0.3465948884570047

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(log_image2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_histModeCount, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_energy_maximum, x1; return_type=Float64)
    tmp4 = safe_call(reduce_biggestAxis, x2; return_type=Float64)
    tmp5 = safe_call(glcm_glcm_entropy_minimum, x2, tmp4; return_type=Float64)
    tmp6 = safe_call(max_img2D, x1, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(mult_img2D, tmp6, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(region_entropy_10p, tmp7, tmp5, tmp3; return_type=Float64)
    tmp9 = safe_call(reduce_length, x3; return_type=Float64)
    tmp10 = safe_call(tophat_2D, x3, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(ando4m_image2D, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(fastscanning_image2D, tmp11, tmp5; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp13 = safe_call(glcm_energy_std, tmp12, tmp8; return_type=Float64)
    tmp14 = safe_call(number_minus, tmp13, tmp2; return_type=Float64)
    tmp15 = safe_call(reduce_std, x2; return_type=Float64)
    tmp16 = safe_call(power_of, tmp15, tmp14; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
