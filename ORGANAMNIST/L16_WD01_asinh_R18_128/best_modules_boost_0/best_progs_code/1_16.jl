# module_key=1_16
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_16/L16_WD01_asinh_R18_128__2/checkpoint_0.pickle
# best_subproblem_loss=0.3160711121294959

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_glcm_var_ref_sum, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_glcm_entropy_minimum, x1; return_type=Float64)
    tmp4 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(region_median_10p, tmp4, tmp3, tmp2; return_type=Float64)
    tmp6 = safe_call(binarize_minimumintermodes2D, ; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(gaussian25_image2D, tmp6, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(glcm_glcm_var_ref_maximum, tmp7, tmp3; return_type=Float64)
    tmp9 = safe_call(moffat25_image2D, x1, tmp3, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(glcm_IDM_std, tmp9; return_type=Float64)
    tmp11 = safe_call(mult_img2D, x1, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(tophat_2D, tmp11; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(gaussian5_image2D, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(glcm_glcm_mean_ref_minimum, tmp13, tmp10; return_type=Float64)
    tmp15 = safe_call(closing_2D, x1, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(minpool_blocks, tmp15, tmp14; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(region_sum_20p, tmp16, tmp8, tmp5; return_type=Float64)
    tmp18 = safe_call(log10_, tmp17; return_type=Float64)
    tmp19 = safe_call(reduce_biggestAxis, tmp9; return_type=Float64)
    tmp20 = safe_call(prewittm_image2D, tmp9, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp21 = safe_call(haar_tb, tmp20, tmp19, tmp10; return_type=Float64)
    tmp22 = safe_call(number_mult, tmp21, tmp18; return_type=Float64)
    tmp23 = safe_call(identity_float, tmp22; return_type=Float64)
    out1 = tmp23
    return (out1,)
end
