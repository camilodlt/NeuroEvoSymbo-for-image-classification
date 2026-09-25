# module_key=1_9
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_9/L16_WD01_asinh_R18_128__3/checkpoint_0.pickle
# best_subproblem_loss=0.18356737227845354

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(glcm_glcm_var_ref_maximum, x1; return_type=Float64)
    tmp2 = safe_call(sobelx_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(reduce_maximum, tmp2; return_type=Float64)
    tmp4 = safe_call(gaussian5_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(experimental_standardize_2D, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(identity_image2D, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(region_sum_10p, tmp6, tmp3, tmp1; return_type=Float64)
    tmp8 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp9 = safe_call(glcm_contrast_std, tmp8; return_type=Float64)
    tmp10 = safe_call(maxpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(glcm_glcm_entropy_std, tmp10, tmp9, tmp7; return_type=Float64)
    tmp12 = safe_call(glcm_energy_mean, x1; return_type=Float64)
    tmp13 = safe_call(findlocalmaxima_image2D, x1, tmp12; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(ando3x_image2D, tmp13, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(glcm_max_prob_sum, tmp14, tmp12; return_type=Float64)
    tmp16 = safe_call(glcm_max_prob_minimum, tmp10, tmp7; return_type=Float64)
    tmp17 = safe_call(sobelx_image2D, tmp10, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(glcm_glcm_entropy_std, tmp17; return_type=Float64)
    tmp19 = safe_call(region_entropy_20p, tmp8, tmp18, tmp16; return_type=Float64)
    tmp20 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp21 = safe_call(glcm_glcm_var_ref_minimum, tmp20; return_type=Float64)
    tmp22 = safe_call(region_min_5p, x1, tmp21, tmp19; return_type=Float64)
    tmp23 = safe_call(number_mult, tmp22, tmp15; return_type=Float64)
    tmp24 = safe_call(findlocalmaxima_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp25 = safe_call(bothat_2D, tmp24; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp26 = safe_call(glcm_correlation_std, tmp25, tmp23, tmp11; return_type=Float64)
    tmp27 = safe_call(identity_float, tmp26; return_type=Float64)
    out1 = tmp27
    return (out1,)
end
