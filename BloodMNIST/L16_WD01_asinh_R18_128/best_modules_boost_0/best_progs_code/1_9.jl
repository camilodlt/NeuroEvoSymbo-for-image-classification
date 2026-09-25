# module_key=1_9
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_9/L16_WD01_asinh_R18_128__2/checkpoint_0.pickle
# best_subproblem_loss=0.12684945734381925

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_contrast_maximum, x1; return_type=Float64)
    tmp2 = safe_call(pi_, ; return_type=Float64)
    tmp3 = safe_call(max_img2D, x2, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(reduce_length, tmp3; return_type=Float64)
    tmp5 = safe_call(binarize_entropy2D, x2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(sobelx_image2D, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(subtract_img2D, x1, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(min_img2D, tmp7, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(glcm_dissimilarity_minimum, tmp8, tmp4, tmp2; return_type=Float64)
    tmp10 = safe_call(glcm_correlation_maximum, x3; return_type=Float64)
    tmp11 = safe_call(argmaxcountpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(glcm_ASM_minimum, tmp11; return_type=Float64)
    tmp13 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp14 = safe_call(zeros_2D, tmp13; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp15 = safe_call(zeros_2D, tmp14; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp16 = safe_call(glcm_glcm_entropy_minimum, tmp15; return_type=Float64)
    tmp17 = safe_call(gaussian25_image2D, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(binarize_sauvola2D, x3, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp19 = safe_call(minpool_blocks, tmp18; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(erosion_2D, tmp19; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp21 = safe_call(moffat5_image2D, tmp20; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp22 = safe_call(watershed_image2D, tmp21, tmp17, tmp16; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp23 = safe_call(glcm_IDM_maximum, tmp22, tmp12, tmp10; return_type=Float64)
    tmp24 = safe_call(number_minus, tmp23, tmp9; return_type=Float64)
    tmp25 = safe_call(safe_div, tmp24, tmp1; return_type=Float64)
    tmp26 = safe_call(identity_float, tmp25; return_type=Float64)
    out1 = tmp26
    return (out1,)
end
