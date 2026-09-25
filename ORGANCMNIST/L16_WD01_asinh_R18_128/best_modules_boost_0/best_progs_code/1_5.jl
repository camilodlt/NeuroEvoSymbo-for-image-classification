# module_key=1_5
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_5/L16_WD01_asinh_R18_128__5/checkpoint_0.pickle
# best_subproblem_loss=0.2224759363446137

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(orientation_energy_135, x1; return_type=Float64)
    tmp2 = safe_call(ando5m_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(orientation_energy_45, tmp2; return_type=Float64)
    tmp4 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(zeros_2D, tmp4; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(glcm_dissimilarity_std, tmp5, tmp3; return_type=Float64)
    tmp7 = safe_call(experimental_tobinary_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(scharry_image2D, tmp7, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(ando5y_image2D, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(ando5y_image2D, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(closing_2D, tmp10, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(ando4x_image2D, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(zeros_2D, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(glcm_glcm_entropy_std, tmp13; return_type=Float64)
    tmp15 = safe_call(reduce_nColors, tmp7; return_type=Float64)
    tmp16 = safe_call(ando5x_image2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(region_mean_20p, tmp16, tmp15, tmp14; return_type=Float64)
    tmp18 = safe_call(number_div, tmp3, tmp1; return_type=Float64)
    tmp19 = safe_call(power_of, tmp18, tmp17; return_type=Float64)
    tmp20 = safe_call(glcm_ASM_sum, tmp2; return_type=Float64)
    tmp21 = safe_call(identity_image2D, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp22 = safe_call(morphogradient_2D, tmp21, tmp20; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp23 = safe_call(watershed_image2D, tmp22, tmp8, tmp15; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp24 = safe_call(region_sum_5p, tmp23, tmp19, tmp1; return_type=Float64)
    tmp25 = safe_call(identity_float, tmp24; return_type=Float64)
    out1 = tmp25
    return (out1,)
end
