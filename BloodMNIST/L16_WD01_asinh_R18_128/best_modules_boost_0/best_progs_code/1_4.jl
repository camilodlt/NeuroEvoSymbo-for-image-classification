# module_key=1_4
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_4/L16_WD01_asinh_R18_128__4/checkpoint_0.pickle
# best_subproblem_loss=0.1531598440633417

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(orientation_energy_0, x2; return_type=Float64)
    tmp2 = safe_call(morpholaplace_2D, x3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(min_img2D, x3, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(log_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(reduce_nColors, tmp4; return_type=Float64)
    tmp6 = safe_call(experimental_normalize_2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(add_img2D, tmp6, x2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp9 = safe_call(glcm_ASM_sum, tmp8; return_type=Float64)
    tmp10 = safe_call(exp_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(ando4m_image2D, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp12 = safe_call(bothat_2D, tmp11, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(max_img2D, tmp12, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(medianpool, tmp13, tmp5, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(reduce_propBlack, tmp14; return_type=Float64)
    tmp16 = safe_call(identity_float, tmp15; return_type=Float64)
    out1 = tmp16
    return (out1,)
end
