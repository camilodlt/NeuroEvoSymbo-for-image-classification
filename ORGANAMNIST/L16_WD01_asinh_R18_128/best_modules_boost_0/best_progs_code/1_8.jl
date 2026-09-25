# module_key=1_8
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_8/L16_WD01_asinh_R18_128__102/checkpoint_0.pickle
# best_subproblem_loss=0.2691154664184183

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(add_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(tophat_2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(glcm_energy_mean, tmp2; return_type=Float64)
    tmp4 = safe_call(ret_1, ; return_type=Float64)
    tmp5 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(glcm_IDM_mean, tmp5; return_type=Float64)
    tmp7 = safe_call(minpool_blocks, x1, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(glcm_IDM_maximum, tmp7; return_type=Float64)
    tmp9 = safe_call(sobelm_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(bothat_2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(glcm_contrast_sum, tmp10; return_type=Float64)
    tmp12 = safe_call(binarize_niblack2D, tmp10, tmp11; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(gaussian9_image2D, tmp12, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(closing_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(subtract_img2D, tmp14, tmp13; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp16 = safe_call(haar_tb, tmp15, tmp4, tmp3; return_type=Float64)
    tmp17 = safe_call(log_, tmp16; return_type=Float64)
    tmp18 = safe_call(identity_float, tmp17; return_type=Float64)
    out1 = tmp18
    return (out1,)
end
