# module_key=1_6
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_6/L16_WD01_asinh_R18_128__3/checkpoint_0.pickle
# best_subproblem_loss=0.1902264523174737

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(mult_img2D, x1, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(argmaxcountpool, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(add_img2D, tmp2, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(glcm_ASM_maximum, x1; return_type=Float64)
    tmp5 = safe_call(gaussian13_image2D, x1, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(ando5y_image2D, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(ando4m_image2D, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(moffat25_image2D, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(min_img2D, tmp8, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(orientation_energy_0, tmp9; return_type=Float64)
    tmp11 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(glcm_ASM_minimum, tmp11; return_type=Float64)
    tmp13 = safe_call(max_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(ando3y_image2D, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(region_mean_20p, tmp14, tmp12, tmp12; return_type=Float64)
    tmp16 = safe_call(experimental_tobinary_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp17 = safe_call(region_sum, tmp16, tmp15, tmp4; return_type=Float64)
    tmp18 = safe_call(closing_2D, tmp2, tmp17; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(experimental_standardize_2D, tmp18; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(orientation_energy_45, tmp19; return_type=Float64)
    tmp21 = safe_call(number_div, tmp20, tmp10; return_type=Float64)
    tmp22 = safe_call(identity_float, tmp21; return_type=Float64)
    out1 = tmp22
    return (out1,)
end
