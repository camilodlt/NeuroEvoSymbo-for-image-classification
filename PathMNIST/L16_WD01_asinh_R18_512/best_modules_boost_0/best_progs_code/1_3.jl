# module_key=1_3
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_3/L16_WD01_asinh_R18_512__3/checkpoint_0.pickle
# best_subproblem_loss=0.31627441708745774

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_IDM_maximum, x1; return_type=Float64)
    tmp2 = safe_call(sobelm_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(morphogradient_2D, tmp2, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(gaussian5_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(grad_orientation, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(min_img2D, tmp5, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(reduce_histModeCount, tmp6; return_type=Float64)
    tmp8 = safe_call(ando4x_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(sobely_image2D, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(reduce_smallerAxis, tmp9; return_type=Float64)
    tmp11 = safe_call(glcm_max_prob_minimum, tmp4, tmp10, tmp7; return_type=Float64)
    tmp12 = safe_call(reduce_minimum, tmp8; return_type=Float64)
    tmp13 = safe_call(erosion_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(add_img2D, x2, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(mult_img2D, tmp14, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp16 = safe_call(bothat_2D, tmp15, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(glcm_correlation_std, tmp16, tmp11; return_type=Float64)
    tmp18 = safe_call(identity_float, tmp17; return_type=Float64)
    out1 = tmp18
    return (out1,)
end
