# module_key=1_9
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_9/L16_WD01_asinh_R18_128__8/checkpoint_0.pickle
# best_subproblem_loss=0.10965728310796852

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(experimental_tobinary_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(morphogradient_2D, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(add_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(mult_img2D, tmp3, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(fastscanning_image2D, tmp4; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(reduce_nColors, tmp5; return_type=Float64)
    tmp7 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp8 = safe_call(glcm_contrast_std, tmp7; return_type=Float64)
    tmp9 = safe_call(glcm_glcm_mean_ref_std, x1; return_type=Float64)
    tmp10 = safe_call(region_max_20p, x1, tmp9, tmp9; return_type=Float64)
    tmp11 = safe_call(glcm_max_prob_sum, x1; return_type=Float64)
    tmp12 = safe_call(fastscanning_image2D, tmp3; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp13 = safe_call(glcm_dissimilarity_maximum, tmp12, tmp11, tmp10; return_type=Float64)
    tmp14 = safe_call(ando3y_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(bickleym_image2D, tmp14; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(gaussian25_image2D, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(avgpool_cross_blocks, tmp16, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(region_max_10p, tmp17, tmp10, tmp8; return_type=Float64)
    tmp19 = safe_call(power_of, tmp18, tmp6; return_type=Float64)
    tmp20 = safe_call(identity_float, tmp19; return_type=Float64)
    out1 = tmp20
    return (out1,)
end
