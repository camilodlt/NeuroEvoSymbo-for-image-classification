# module_key=1_12
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_12/L16_WD01_asinh_R18_128__4/checkpoint_0.pickle
# best_subproblem_loss=0.36828759236088326

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(closing_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(dominant_orientation, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_energy_std, x1; return_type=Float64)
    tmp4 = safe_call(safe_div, tmp3, tmp2; return_type=Float64)
    tmp5 = safe_call(binarize_sauvola2D, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(watershed_image2D, tmp5, tmp3; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp7 = safe_call(glcm_max_prob_std, tmp6; return_type=Float64)
    tmp8 = safe_call(reduce_maximum, x1; return_type=Float64)
    tmp9 = safe_call(gaussian5_image2D, tmp1, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(ando5y_image2D, tmp5, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(max_img2D, x1, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp12 = safe_call(subtract_img2D, tmp11, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(gaussian13_image2D, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(glcm_max_prob_minimum, tmp13, tmp7, tmp4; return_type=Float64)
    tmp15 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp16 = safe_call(glcm_correlation_sum, tmp15; return_type=Float64)
    tmp17 = safe_call(minpool, tmp12, tmp16; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(experimental_standardize_2D, tmp17; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(closing_2D, tmp18, tmp14; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(reduce_propBlack, tmp19; return_type=Float64)
    tmp21 = safe_call(identity_float, tmp20; return_type=Float64)
    out1 = tmp21
    return (out1,)
end
