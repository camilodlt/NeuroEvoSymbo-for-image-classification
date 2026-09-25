# module_key=1_12
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_12/L16_WD01_asinh_R18_512__101/checkpoint_0.pickle
# best_subproblem_loss=0.30293267791541967

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(fastscanning_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_ASM_std, tmp1; return_type=Float64)
    tmp3 = safe_call(reduce_histMode, x3; return_type=Float64)
    tmp4 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_contrast_maximum, tmp4, tmp3; return_type=Float64)
    tmp6 = safe_call(glcm_glcm_mean_ref_maximum, x2; return_type=Float64)
    tmp7 = safe_call(argmincountpool, x2, tmp6, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(glcm_glcm_mean_ref_minimum, tmp7; return_type=Float64)
    tmp9 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_contrast_maximum, tmp9; return_type=Float64)
    tmp11 = safe_call(gaussian5_image2D, x1, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(medianpool, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(argmaxcountpool, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(subtract_img2D, tmp13, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(prewittm_image2D, tmp14, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(argmaxcountpool, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(morpholaplace_2D, tmp16, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(reduce_propWhite, tmp17; return_type=Float64)
    tmp19 = safe_call(bothat_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(orientation_coherence, tmp19; return_type=Float64)
    tmp21 = safe_call(number_minus, tmp20, tmp18; return_type=Float64)
    tmp22 = safe_call(identity_float, tmp21; return_type=Float64)
    out1 = tmp22
    return (out1,)
end
