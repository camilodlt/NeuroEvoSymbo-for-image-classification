# module_key=1_14
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_14/L16_WD01_asinh_R18_512__2/checkpoint_0.pickle
# best_subproblem_loss=0.2644138002597691

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(avgpool_blocks, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_maximum, tmp1; return_type=Float64)
    tmp3 = safe_call(opening_2D, x1, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(glcm_max_prob_sum, tmp3; return_type=Float64)
    tmp5 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(glcm_dissimilarity_std, tmp5; return_type=Float64)
    tmp7 = safe_call(number_sum, tmp6, tmp4; return_type=Float64)
    tmp8 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp9 = safe_call(glcm_contrast_minimum, tmp8; return_type=Float64)
    tmp10 = safe_call(experimental_not, tmp9; return_type=Float64)
    tmp11 = safe_call(glcm_max_prob_mean, x2; return_type=Float64)
    tmp12 = safe_call(glcm_max_prob_minimum, x2, tmp11; return_type=Float64)
    tmp13 = safe_call(erosion_2D, x2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(experimental_tosegment_image2D, tmp13; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp15 = safe_call(glcm_glcm_mean_ref_minimum, tmp14, tmp12, tmp10; return_type=Float64)
    tmp16 = safe_call(number_div, tmp15, tmp7; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
