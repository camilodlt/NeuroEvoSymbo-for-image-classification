# module_key=1_7
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_7/L16_WD01_asinh_R18_128__101/checkpoint_0.pickle
# best_subproblem_loss=0.25993331660741836

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(dilation_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_minimum, tmp1; return_type=Float64)
    tmp3 = safe_call(powerof_image2D, x1, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(log_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(reduce_histMode, tmp4; return_type=Float64)
    tmp6 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp7 = safe_call(glcm_glcm_entropy_minimum, tmp6; return_type=Float64)
    tmp8 = safe_call(reduce_mean, x1; return_type=Float64)
    tmp9 = safe_call(gaussian9_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(iqrpool, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(haar_lr, tmp10, tmp8, tmp8; return_type=Float64)
    tmp12 = safe_call(bickleyx_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(ando5m_image2D, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(subtract_img2D, x1, tmp13; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(region_sum_20p, tmp14, tmp11, tmp7; return_type=Float64)
    tmp16 = safe_call(fastscanning_image2D, tmp4; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp17 = safe_call(glcm_dissimilarity_sum, tmp16, tmp15; return_type=Float64)
    tmp18 = safe_call(reduce_maximum, x1; return_type=Float64)
    tmp19 = safe_call(gaussian5_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(region_sum_10p, tmp19, tmp18, tmp17; return_type=Float64)
    tmp21 = safe_call(maxpool_blocks, x1, tmp20; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp22 = safe_call(region_sum, tmp21, tmp5, tmp5; return_type=Float64)
    tmp23 = safe_call(identity_float, tmp22; return_type=Float64)
    out1 = tmp23
    return (out1,)
end
