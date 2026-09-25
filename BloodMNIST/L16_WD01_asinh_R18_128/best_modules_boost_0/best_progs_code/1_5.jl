# module_key=1_5
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_5/L16_WD01_asinh_R18_128__12/checkpoint_0.pickle
# best_subproblem_loss=0.2185097634774551

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(reduce_smallerAxis, x2; return_type=Float64)
    tmp2 = safe_call(bothat_2D, x1, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(morpholaplace_2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(reduce_nColors, tmp3; return_type=Float64)
    tmp5 = safe_call(opening_2D, x2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(add_img2D, tmp5, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(reduce_nColors, tmp6; return_type=Float64)
    tmp8 = safe_call(argmaxcountpool, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(glcm_correlation_maximum, tmp8; return_type=Float64)
    tmp10 = safe_call(binarize_sauvola2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(reduce_length, tmp10; return_type=Float64)
    tmp12 = safe_call(maxpool_blocks, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(erosion_2D, tmp12, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(glcm_ASM_sum, tmp13, tmp11, tmp9; return_type=Float64)
    tmp15 = safe_call(dilation_2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(opening_2D, tmp15; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp17 = safe_call(watershed_image2D, tmp16, tmp14; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp18 = safe_call(glcm_IDM_sum, tmp17; return_type=Float64)
    tmp19 = safe_call(erosion_2D, tmp13, tmp18; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(watershed_image2D, tmp19; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp21 = safe_call(glcm_glcm_var_ref_mean, tmp20; return_type=Float64)
    tmp22 = safe_call(safe_div, tmp21, tmp7; return_type=Float64)
    tmp23 = safe_call(safe_div, tmp22, tmp4; return_type=Float64)
    tmp24 = safe_call(identity_float, tmp23; return_type=Float64)
    out1 = tmp24
    return (out1,)
end
