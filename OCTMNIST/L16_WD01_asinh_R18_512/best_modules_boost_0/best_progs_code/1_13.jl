# module_key=1_13
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_13/L16_WD01_asinh_R18_512__3/checkpoint_0.pickle
# best_subproblem_loss=0.2945722301861303

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_correlation_sum, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_contrast_maximum, x1; return_type=Float64)
    tmp4 = safe_call(moffat25_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(glcm_glcm_var_ref_minimum, tmp4, tmp3, tmp2; return_type=Float64)
    tmp6 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp7 = safe_call(glcm_dissimilarity_maximum, tmp6; return_type=Float64)
    tmp8 = safe_call(glcm_correlation_minimum, x1, tmp7, tmp5; return_type=Float64)
    tmp9 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_contrast_mean, tmp9, tmp8; return_type=Float64)
    tmp11 = safe_call(binarize_yen2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp12 = safe_call(dog_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(add_img2D, tmp12, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(orientation_energy_0, tmp13; return_type=Float64)
    tmp15 = safe_call(bothat_2D, x1, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(gaussian9_image2D, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(dog_image2D, tmp16, tmp14, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(orientation_spread, tmp17; return_type=Float64)
    tmp19 = safe_call(identity_float, tmp18; return_type=Float64)
    out1 = tmp19
    return (out1,)
end
