# module_key=1_8
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_8/L16_WD01_asinh_R18_128__8/checkpoint_0.pickle
# best_subproblem_loss=0.1957984793434564

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(glcm_IDM_maximum, x1; return_type=Float64)
    tmp2 = safe_call(glcm_contrast_maximum, x1; return_type=Float64)
    tmp3 = safe_call(ando5y_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(dilation_2D, tmp3, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(orientation_energy_0, tmp4; return_type=Float64)
    tmp6 = safe_call(glcm_glcm_var_ref_sum, tmp3; return_type=Float64)
    tmp7 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp8 = safe_call(glcm_correlation_maximum, tmp7, tmp6, tmp5; return_type=Float64)
    tmp9 = safe_call(experimental_tobinary_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(erosion_2D, tmp9, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(haar_diag_main, tmp10, tmp1, tmp1; return_type=Float64)
    tmp12 = safe_call(erosion_2D, tmp9, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(dilation_2D, tmp4, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(moffat13_image2D, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(add_img2D, tmp14, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(bsubtract_image2D, tmp15, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(region_min_20p, tmp16, tmp6, tmp11; return_type=Float64)
    tmp18 = safe_call(reduce_length, tmp9; return_type=Float64)
    tmp19 = safe_call(reduce_histMode, x1; return_type=Float64)
    tmp20 = safe_call(ando3x_image2D, tmp10, tmp19; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp21 = safe_call(gaussian25_image2D, tmp4, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp22 = safe_call(max_img2D, tmp21, tmp20; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp23 = safe_call(haar_center_surround, tmp22, tmp6, tmp18; return_type=Float64)
    tmp24 = safe_call(number_minus, tmp23, tmp17; return_type=Float64)
    tmp25 = safe_call(identity_float, tmp24; return_type=Float64)
    out1 = tmp25
    return (out1,)
end
