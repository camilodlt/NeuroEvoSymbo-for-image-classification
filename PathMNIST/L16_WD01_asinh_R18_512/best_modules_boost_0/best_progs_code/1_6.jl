# module_key=1_6
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_6/L16_WD01_asinh_R18_512__103/checkpoint_0.pickle
# best_subproblem_loss=0.30857599107848943

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(mult_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(bothat_2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(bothat_2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(bickleyx_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(glcm_dissimilarity_sum, tmp4; return_type=Float64)
    tmp6 = safe_call(glcm_glcm_var_ref_sum, x1; return_type=Float64)
    tmp7 = safe_call(erosion_2D, x1, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(bickleyx_image2D, tmp7, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(glcm_IDM_mean, tmp8, tmp5, tmp5; return_type=Float64)
    tmp10 = safe_call(gaussian5_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(glcm_energy_sum, tmp10; return_type=Float64)
    tmp12 = safe_call(number_div, tmp11, tmp9; return_type=Float64)
    tmp13 = safe_call(identity_float, tmp12; return_type=Float64)
    out1 = tmp13
    return (out1,)
end
