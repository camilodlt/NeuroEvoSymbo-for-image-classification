# module_key=1_11
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_11/L16_WD01_asinh_R18_128__2/checkpoint_0.pickle
# best_subproblem_loss=0.24196453449980948

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_correlation_std, tmp1; return_type=Float64)
    tmp3 = safe_call(ando4y_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(fastscanning_image2D, tmp3; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_max_prob_maximum, tmp4; return_type=Float64)
    tmp6 = safe_call(haar_tb, x1, tmp2, tmp5; return_type=Float64)
    tmp7 = safe_call(meanpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(min_img2D, tmp7, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(region_sum_20p, tmp8, tmp6, tmp2; return_type=Float64)
    tmp10 = safe_call(glcm_max_prob_sum, x1; return_type=Float64)
    tmp11 = safe_call(findlocalminima_image2D, tmp7, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp12 = safe_call(fastscanning_image2D, tmp11; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp13 = safe_call(glcm_max_prob_minimum, tmp12, tmp2; return_type=Float64)
    tmp14 = safe_call(meanpool, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(max_img2D, tmp14, tmp14; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(max_img2D, tmp15, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp17 = safe_call(region_sum_5p, tmp16, tmp13, tmp9; return_type=Float64)
    tmp18 = safe_call(dilation_2D, tmp11; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp19 = safe_call(findlocalminima_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(if_else_multiplexer, tmp9, tmp19, tmp18; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp21 = safe_call(bickleyx_image2D, tmp20, tmp17; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp22 = safe_call(reduce_nColors, tmp21; return_type=Float64)
    tmp23 = safe_call(identity_float, tmp22; return_type=Float64)
    out1 = tmp23
    return (out1,)
end
