# module_key=1_1
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_1/L16_WD01_asinh_R18_128__3/checkpoint_0.pickle
# best_subproblem_loss=0.21432542338499494

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(fastscanning_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(zeros_2D, tmp1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp3 = safe_call(glcm_contrast_minimum, tmp2; return_type=Float64)
    tmp4 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_dissimilarity_mean, tmp4, tmp3; return_type=Float64)
    tmp6 = safe_call(glcm_contrast_std, x1; return_type=Float64)
    tmp7 = safe_call(binarize_moments2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(opening_2D, tmp7, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(reduce_propBlack, tmp8; return_type=Float64)
    tmp10 = safe_call(uniquecountpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(sobelx_image2D, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(prewitty_image2D, tmp11, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(dominant_orientation, tmp12; return_type=Float64)
    tmp14 = safe_call(experimental_not, tmp13; return_type=Float64)
    tmp15 = safe_call(gaussian25_image2D, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(add_img2D, tmp15, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(bmult_image2D, tmp16, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(region_sum_10p, tmp17, tmp14, tmp5; return_type=Float64)
    tmp19 = safe_call(identity_float, tmp18; return_type=Float64)
    out1 = tmp19
    return (out1,)
end
