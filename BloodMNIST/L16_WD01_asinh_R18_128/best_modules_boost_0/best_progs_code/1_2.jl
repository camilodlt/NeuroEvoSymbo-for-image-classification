# module_key=1_2
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_2/L16_WD01_asinh_R18_128__4/checkpoint_0.pickle
# best_subproblem_loss=0.193292395482195

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(binarize_minimumintermodes2D, ; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(reduce_smallerAxis, tmp1; return_type=Float64)
    tmp3 = safe_call(reduce_length, x3; return_type=Float64)
    tmp4 = safe_call(add_img2D, x1, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(ando3m_image2D, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(exp_image2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(gaussian9_image2D, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(binarize_yen2D, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(morphogradient_2D, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(max_img2D, tmp9, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(haar_center_surround, tmp10, tmp3, tmp2; return_type=Float64)
    tmp12 = safe_call(identity_float, tmp11; return_type=Float64)
    out1 = tmp12
    return (out1,)
end
