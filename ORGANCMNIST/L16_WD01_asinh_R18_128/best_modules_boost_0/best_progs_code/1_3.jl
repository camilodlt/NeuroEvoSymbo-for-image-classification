# module_key=1_3
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_3/L16_WD01_asinh_R18_128__1/checkpoint_0.pickle
# best_subproblem_loss=0.12571176425408837

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(erosion_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(argmincountpool, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(grad_orientation, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(reduce_median, tmp3; return_type=Float64)
    tmp5 = safe_call(glcm_ASM_mean, x1; return_type=Float64)
    tmp6 = safe_call(experimental_tobinary_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(watershed_image2D, tmp6, tmp6, tmp5; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp8 = safe_call(zeros_2D, tmp7; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp9 = safe_call(glcm_dissimilarity_mean, tmp8; return_type=Float64)
    tmp10 = safe_call(fastscanning_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp11 = safe_call(identity_image2D, tmp10; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(reduce_biggestAxis, tmp11; return_type=Float64)
    tmp13 = safe_call(subtract_img2D, x1, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(haar_diag_main, tmp13, tmp12, tmp9; return_type=Float64)
    tmp15 = safe_call(glcm_dissimilarity_minimum, tmp7; return_type=Float64)
    tmp16 = safe_call(prewittm_image2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(bsubtract_image2D, tmp16, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp19 = safe_call(glcm_energy_maximum, tmp18; return_type=Float64)
    tmp20 = safe_call(binarize_moments2D, x1, tmp19; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp21 = safe_call(gaussian13_image2D, tmp20; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp22 = safe_call(mult_img2D, tmp21, tmp17; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp23 = safe_call(region_mean, tmp22, tmp5, tmp14; return_type=Float64)
    tmp24 = safe_call(number_sum, tmp23, tmp4; return_type=Float64)
    tmp25 = safe_call(identity_float, tmp24; return_type=Float64)
    out1 = tmp25
    return (out1,)
end
