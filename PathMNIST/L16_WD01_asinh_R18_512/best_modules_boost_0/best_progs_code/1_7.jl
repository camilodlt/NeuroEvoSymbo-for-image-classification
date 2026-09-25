# module_key=1_7
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_7/L16_WD01_asinh_R18_512__2/checkpoint_0.pickle
# best_subproblem_loss=0.3146492878932652

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_dissimilarity_maximum, x1; return_type=Float64)
    tmp2 = safe_call(glcm_energy_minimum, x2; return_type=Float64)
    tmp3 = safe_call(number_minus, tmp2, tmp1; return_type=Float64)
    tmp4 = safe_call(exp_image2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(erosion_2D, tmp4, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(experimental_standardize_2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(ando4m_image2D, tmp6, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(max_img2D, tmp7, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(dilation_2D, tmp8, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(reduce_propWhite, tmp9; return_type=Float64)
    tmp11 = safe_call(identity_float, tmp10; return_type=Float64)
    out1 = tmp11
    return (out1,)
end
