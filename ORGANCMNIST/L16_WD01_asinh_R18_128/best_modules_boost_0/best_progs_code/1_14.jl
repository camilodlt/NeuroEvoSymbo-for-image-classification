# module_key=1_14
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_14/L16_WD01_asinh_R18_128__6/checkpoint_0.pickle
# best_subproblem_loss=0.1239520116434657

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(erosion_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(avgpool_blocks, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(zeros_2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(glcm_correlation_std, tmp3; return_type=Float64)
    tmp5 = safe_call(binarize_otsu2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(reduce_propWhite, tmp5; return_type=Float64)
    tmp7 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(region_max_20p, tmp7, tmp6, tmp4; return_type=Float64)
    tmp9 = safe_call(ando3x_image2D, tmp1, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(orientation_spread, tmp9; return_type=Float64)
    tmp11 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(glcm_glcm_mean_ref_std, tmp11; return_type=Float64)
    tmp13 = safe_call(opening_2D, tmp2, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(glcm_contrast_minimum, tmp13; return_type=Float64)
    tmp15 = safe_call(region_sum_20p, tmp2, tmp6, tmp14; return_type=Float64)
    tmp16 = safe_call(glcm_IDM_std, tmp11; return_type=Float64)
    tmp17 = safe_call(grad_orientation, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(binarize_sauvola2D, tmp17, tmp6, tmp16; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp19 = safe_call(fastscanning_image2D, tmp18; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp20 = safe_call(glcm_max_prob_minimum, tmp19, tmp15, tmp15; return_type=Float64)
    tmp21 = safe_call(safe_div, tmp14, tmp20; return_type=Float64)
    tmp22 = safe_call(safe_div, tmp21, tmp10; return_type=Float64)
    tmp23 = safe_call(identity_float, tmp22; return_type=Float64)
    out1 = tmp23
    return (out1,)
end
