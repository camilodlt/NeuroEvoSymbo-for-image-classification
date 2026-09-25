# module_key=1_16
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_16/L16_WD01_asinh_R18_128__10/checkpoint_0.pickle
# best_subproblem_loss=0.29206561011734433

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(closing_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_std, tmp1; return_type=Float64)
    tmp3 = safe_call(reduce_length, x2; return_type=Float64)
    tmp4 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_contrast_minimum, tmp4; return_type=Float64)
    tmp6 = safe_call(avgpool_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(opening_2D, tmp6, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(dilation_2D, tmp7, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(dilation_2D, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(glcm_dissimilarity_std, tmp9, tmp2; return_type=Float64)
    tmp11 = safe_call(identity_image2D, tmp4; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(glcm_max_prob_std, tmp11, tmp5, tmp5; return_type=Float64)
    tmp13 = safe_call(loginv_image2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(avgpool_blocks, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(add_img2D, tmp14, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(glcm_correlation_minimum, tmp15, tmp12, tmp12; return_type=Float64)
    tmp17 = safe_call(modulo, tmp16, tmp10; return_type=Float64)
    tmp18 = safe_call(identity_float, tmp17; return_type=Float64)
    out1 = tmp18
    return (out1,)
end
