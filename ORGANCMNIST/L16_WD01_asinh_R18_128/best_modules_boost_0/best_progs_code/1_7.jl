# module_key=1_7
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_7/L16_WD01_asinh_R18_128__8/checkpoint_0.pickle
# best_subproblem_loss=0.26329542050131916

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(binarize_minimumintermodes2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(reduce_length, tmp1; return_type=Float64)
    tmp3 = safe_call(argmaxcountpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(add_img2D, tmp3, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(reduce_biggestAxis, tmp4; return_type=Float64)
    tmp6 = safe_call(scharry_image2D, tmp4, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(ando3m_image2D, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(glcm_correlation_sum, tmp7, tmp2; return_type=Float64)
    tmp9 = safe_call(reduce_std, x1; return_type=Float64)
    tmp10 = safe_call(moffat5_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(ando3m_image2D, tmp10, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(sobelm_image2D, tmp11, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(morphogradient_2D, tmp12, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(orientation_energy_135, tmp13; return_type=Float64)
    tmp15 = safe_call(exp_, tmp5; return_type=Float64)
    tmp16 = safe_call(glcm_glcm_mean_ref_maximum, tmp10; return_type=Float64)
    tmp17 = safe_call(experimental_tobinary_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(watershed_image2D, tmp17, tmp1, tmp16; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp19 = safe_call(region_max_20p, tmp18, tmp16, tmp15; return_type=Float64)
    tmp20 = safe_call(if_else_multiplexer, tmp19, tmp14, tmp8; return_type=Float64)
    tmp21 = safe_call(identity_float, tmp20; return_type=Float64)
    out1 = tmp21
    return (out1,)
end
