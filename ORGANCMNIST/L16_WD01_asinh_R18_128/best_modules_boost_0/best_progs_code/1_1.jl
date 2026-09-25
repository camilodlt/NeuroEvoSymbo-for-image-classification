# module_key=1_1
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_1/L16_WD01_asinh_R18_128__6/checkpoint_0.pickle
# best_subproblem_loss=0.1421776136246311

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_correlation_minimum, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_glcm_var_ref_sum, x1; return_type=Float64)
    tmp4 = safe_call(glcm_glcm_mean_ref_minimum, x1; return_type=Float64)
    tmp5 = safe_call(morphogradient_2D, x1, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(binarize_moments2D, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(morpholaplace_2D, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(orientation_select, tmp7, tmp3, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(binarize_moments2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(stdpool, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(subtract_img2D, x1, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp12 = safe_call(add_img2D, x1, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(subtract_img2D, tmp12, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(subtract_img2D, tmp13, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(reduce_propWhite, tmp14; return_type=Float64)
    tmp16 = safe_call(log_, tmp4; return_type=Float64)
    tmp17 = safe_call(subtract_img2D, tmp12, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(haar_diag_main, tmp17, tmp16, tmp4; return_type=Float64)
    tmp19 = safe_call(gaussian9_image2D, tmp5, tmp18; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(morpholaplace_2D, tmp19; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp21 = safe_call(orientation_energy_135, tmp20; return_type=Float64)
    tmp22 = safe_call(number_sum, tmp21, tmp15; return_type=Float64)
    tmp23 = safe_call(identity_float, tmp22; return_type=Float64)
    out1 = tmp23
    return (out1,)
end
