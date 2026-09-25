# module_key=1_14
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_14/L16_WD01_asinh_R18_128__1/checkpoint_0.pickle
# best_subproblem_loss=0.18948106075181292

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(ando3y_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(glcm_IDM_std, tmp1; return_type=Float64)
    tmp3 = safe_call(erosion_2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(add_img2D, x1, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(bickleyy_image2D, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(max_img2D, tmp3, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_dissimilarity_minimum, tmp6; return_type=Float64)
    tmp8 = safe_call(number_div, tmp7, tmp2; return_type=Float64)
    tmp9 = safe_call(identity_float, tmp8; return_type=Float64)
    out1 = tmp9
    return (out1,)
end
