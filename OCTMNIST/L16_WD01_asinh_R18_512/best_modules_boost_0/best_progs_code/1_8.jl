# module_key=1_8
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_8/L16_WD01_asinh_R18_512__101/checkpoint_0.pickle
# best_subproblem_loss=0.25311872170761807

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(glcm_glcm_mean_ref_sum, x1; return_type=Float64)
    tmp2 = safe_call(mult_img2D, x1, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(sobelx_image2D, tmp2, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(reduce_biggestAxis, tmp3; return_type=Float64)
    tmp5 = safe_call(identity_float, tmp4; return_type=Float64)
    tmp6 = safe_call(glcm_IDM_sum, x1; return_type=Float64)
    tmp7 = safe_call(log_, tmp6; return_type=Float64)
    tmp8 = safe_call(ando4m_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(add_img2D, x1, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(add_img2D, tmp9, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(experimental_standardize_2D, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(bothat_2D, tmp11, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(laplacian3_image2D, tmp12, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(orientation_spread, tmp13; return_type=Float64)
    tmp15 = safe_call(identity_float, tmp14; return_type=Float64)
    out1 = tmp15
    return (out1,)
end
