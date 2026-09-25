# module_key=1_16
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_16/L16_WD01_asinh_R18_512__4/checkpoint_0.pickle
# best_subproblem_loss=0.5366796967493508

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(binarize_niblack2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(scharrm_image2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(sobelm_image2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(identity_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(glcm_correlation_minimum, tmp4; return_type=Float64)
    tmp6 = safe_call(dominant_orientation, x1; return_type=Float64)
    tmp7 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp8 = safe_call(glcm_glcm_var_ref_std, tmp7, tmp6; return_type=Float64)
    tmp9 = safe_call(orientation_select, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(erosion_2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(glcm_contrast_minimum, tmp10, tmp8, tmp5; return_type=Float64)
    tmp12 = safe_call(glcm_max_prob_maximum, x1; return_type=Float64)
    tmp13 = safe_call(log_, tmp12; return_type=Float64)
    tmp14 = safe_call(meanpool, tmp3, tmp13, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(loginv_image2D, tmp14; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(orientation_coherence, tmp15; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
