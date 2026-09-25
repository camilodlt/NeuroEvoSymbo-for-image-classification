# module_key=1_15
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_15/L16_WD01_asinh_R18_128__6/checkpoint_0.pickle
# best_subproblem_loss=0.15019687393701275

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(erosion_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(laplacian3_image2D, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(reduce_propBlack, tmp2; return_type=Float64)
    tmp4 = safe_call(ret_1, ; return_type=Float64)
    tmp5 = safe_call(morpholaplace_2D, tmp1, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(tophat_2D, tmp5, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(orientation_energy_135, tmp6; return_type=Float64)
    tmp8 = safe_call(binarize_moments2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(watershed_image2D, tmp8; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_glcm_entropy_std, tmp9, tmp4, tmp3; return_type=Float64)
    tmp11 = safe_call(morphogradient_2D, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(ando3y_image2D, tmp11, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(orientation_energy_45, tmp12; return_type=Float64)
    tmp14 = safe_call(safe_div, tmp13, tmp7; return_type=Float64)
    tmp15 = safe_call(identity_float, tmp14; return_type=Float64)
    out1 = tmp15
    return (out1,)
end
