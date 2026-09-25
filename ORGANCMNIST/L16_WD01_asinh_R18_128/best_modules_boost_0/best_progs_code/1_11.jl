# module_key=1_11
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_11/L16_WD01_asinh_R18_128__2/checkpoint_0.pickle
# best_subproblem_loss=0.19053956596246924

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(ando5m_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(ando3m_image2D, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(reduce_histModeCount, x1; return_type=Float64)
    tmp4 = safe_call(binarize_niblack2D, x1, tmp3, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(min_img2D, tmp1, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(subtract_img2D, tmp5, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_IDM_maximum, tmp6; return_type=Float64)
    tmp8 = safe_call(findlocalminima_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(watershed_image2D, tmp8; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_IDM_std, tmp9; return_type=Float64)
    tmp11 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(glcm_correlation_maximum, tmp11; return_type=Float64)
    tmp13 = safe_call(bickleyx_image2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(ando3y_image2D, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(bothat_2D, tmp14, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(morphogradient_2D, tmp15, tmp10, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(orientation_energy_45, tmp16; return_type=Float64)
    tmp18 = safe_call(log_, tmp17; return_type=Float64)
    tmp19 = safe_call(identity_float, tmp18; return_type=Float64)
    out1 = tmp19
    return (out1,)
end
