# module_key=1_16
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_16/L16_WD01_asinh_R18_512__3/checkpoint_0.pickle
# best_subproblem_loss=0.30469339209031976

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(dilation_2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(glcm_IDM_std, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_glcm_var_ref_sum, x2; return_type=Float64)
    tmp4 = safe_call(orientation_energy_0, x1; return_type=Float64)
    tmp5 = safe_call(ando5m_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(glcm_max_prob_minimum, tmp5, tmp4; return_type=Float64)
    tmp7 = safe_call(identity_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(region_energy_10p, tmp7, tmp6, tmp3; return_type=Float64)
    tmp9 = safe_call(max_img2D, x1, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(ando5m_image2D, tmp9, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(bothat_2D, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(glcm_energy_maximum, tmp11, tmp2; return_type=Float64)
    tmp13 = safe_call(binarize_niblack2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(experimental_tosegment_image2D, tmp13; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp15 = safe_call(glcm_max_prob_minimum, tmp14; return_type=Float64)
    tmp16 = safe_call(number_div, tmp15, tmp12; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
