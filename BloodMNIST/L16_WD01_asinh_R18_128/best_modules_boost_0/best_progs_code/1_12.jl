# module_key=1_12
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_12/L16_WD01_asinh_R18_128__11/checkpoint_0.pickle
# best_subproblem_loss=0.16022771916434608

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_correlation_mean, tmp1; return_type=Float64)
    tmp3 = safe_call(stdpool, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(glcm_energy_mean, tmp3, tmp2; return_type=Float64)
    tmp5 = safe_call(reduce_maximum, x3; return_type=Float64)
    tmp6 = safe_call(glcm_dissimilarity_std, x3; return_type=Float64)
    tmp7 = safe_call(gaussian13_image2D, x3, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(glcm_ASM_mean, x1, tmp5, tmp5; return_type=Float64)
    tmp9 = safe_call(glcm_glcm_entropy_std, x1; return_type=Float64)
    tmp10 = safe_call(glcm_glcm_mean_ref_sum, x3, tmp9, tmp5; return_type=Float64)
    tmp11 = safe_call(log_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(glcm_energy_std, tmp11; return_type=Float64)
    tmp13 = safe_call(closing_2D, tmp3, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(morphogradient_2D, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(add_img2D, tmp14, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(gaussian5_image2D, tmp15, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp17 = safe_call(closing_2D, tmp16, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(subtract_img2D, tmp17, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp19 = safe_call(haar_center_surround, tmp18, tmp5, tmp4; return_type=Float64)
    tmp20 = safe_call(identity_float, tmp19; return_type=Float64)
    out1 = tmp20
    return (out1,)
end
