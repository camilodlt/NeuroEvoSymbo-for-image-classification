# module_key=1_1
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_1/L16_WD01_asinh_R18_512__102/checkpoint_0.pickle
# best_subproblem_loss=0.348855836009734

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(exp_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(mult_img2D, x2, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(maxpool_cross_blocks, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(reduce_histModeCount, tmp3; return_type=Float64)
    tmp5 = safe_call(tophat_2D, x3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(reduce_smallerAxis, tmp5; return_type=Float64)
    tmp7 = safe_call(experimental_standardize_2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(orientation_coherence, tmp7; return_type=Float64)
    tmp9 = safe_call(tophat_2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(glcm_correlation_mean, tmp9; return_type=Float64)
    tmp11 = safe_call(safe_div, tmp10, tmp8; return_type=Float64)
    tmp12 = safe_call(fastscanning_image2D, x2; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp13 = safe_call(glcm_IDM_mean, tmp12, tmp11, tmp6; return_type=Float64)
    tmp14 = safe_call(safe_div, tmp13, tmp4; return_type=Float64)
    tmp15 = safe_call(identity_float, tmp14; return_type=Float64)
    out1 = tmp15
    return (out1,)
end
