# module_key=1_1
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_1/L16_WD01_asinh_R18_512__3/checkpoint_0.pickle
# best_subproblem_loss=0.20510126196858725

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(zeros_2D, tmp1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp3 = safe_call(glcm_glcm_entropy_maximum, tmp2; return_type=Float64)
    tmp4 = safe_call(glcm_contrast_maximum, tmp1; return_type=Float64)
    tmp5 = safe_call(binarize_niblack2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(closing_2D, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(bickleyx_image2D, tmp6, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(ando3x_image2D, tmp7, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(add_img2D, x1, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(subtract_img2D, tmp9, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(orientation_spread, tmp10; return_type=Float64)
    tmp12 = safe_call(reduce_minimum, x1; return_type=Float64)
    tmp13 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp14 = safe_call(glcm_glcm_var_ref_sum, tmp13, tmp12; return_type=Float64)
    tmp15 = safe_call(bothat_2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(prewittm_image2D, tmp15, tmp14; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(orientation_coherence, tmp16; return_type=Float64)
    tmp18 = safe_call(power_of, tmp17, tmp11; return_type=Float64)
    tmp19 = safe_call(identity_float, tmp18; return_type=Float64)
    out1 = tmp19
    return (out1,)
end
