# module_key=1_15
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_15/L16_WD01_asinh_R18_128__101/checkpoint_0.pickle
# best_subproblem_loss=0.1429835322097237

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(avgpool_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_maximum, tmp1; return_type=Float64)
    tmp3 = safe_call(meanpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(gaussian17_image2D, tmp3, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(dilation_2D, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(experimental_tosegment_image2D, tmp5; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp7 = safe_call(glcm_energy_std, tmp6; return_type=Float64)
    tmp8 = safe_call(glcm_glcm_var_ref_maximum, x1; return_type=Float64)
    tmp9 = safe_call(glcm_correlation_sum, x1; return_type=Float64)
    tmp10 = safe_call(glcm_dissimilarity_sum, tmp1, tmp9, tmp8; return_type=Float64)
    tmp11 = safe_call(glcm_dissimilarity_mean, tmp1, tmp10; return_type=Float64)
    tmp12 = safe_call(binarize_moments2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(add_img2D, tmp12, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(mult_img2D, tmp3, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(orientation_select, tmp14, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(binarize_yen2D, tmp15; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp17 = safe_call(haar_tb, tmp16, tmp11, tmp7; return_type=Float64)
    tmp18 = safe_call(fastscanning_image2D, tmp3; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp19 = safe_call(reduce_nColors, tmp18; return_type=Float64)
    tmp20 = safe_call(number_mult, tmp19, tmp17; return_type=Float64)
    tmp21 = safe_call(identity_float, tmp20; return_type=Float64)
    out1 = tmp21
    return (out1,)
end
