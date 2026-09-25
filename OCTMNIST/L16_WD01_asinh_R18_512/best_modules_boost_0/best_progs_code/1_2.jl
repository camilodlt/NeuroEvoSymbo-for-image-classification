# module_key=1_2
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_2/L16_WD01_asinh_R18_512__2/checkpoint_0.pickle
# best_subproblem_loss=0.2710733913175446

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(minpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(glcm_max_prob_minimum, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_max_prob_mean, x1; return_type=Float64)
    tmp4 = safe_call(max_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(dilation_2D, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(watershed_image2D, tmp5; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp7 = safe_call(region_sum, tmp6, tmp3, tmp2; return_type=Float64)
    tmp8 = safe_call(tophat_2D, x1, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(orientation_energy_90, tmp8; return_type=Float64)
    tmp10 = safe_call(glcm_glcm_mean_ref_minimum, x1; return_type=Float64)
    tmp11 = safe_call(prewitty_image2D, x1, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(glcm_contrast_minimum, tmp11; return_type=Float64)
    tmp13 = safe_call(log_, tmp12; return_type=Float64)
    tmp14 = safe_call(opening_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(bothat_2D, tmp14, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(orientation_coherence, tmp15; return_type=Float64)
    tmp17 = safe_call(number_minus, tmp16, tmp9; return_type=Float64)
    tmp18 = safe_call(identity_float, tmp17; return_type=Float64)
    out1 = tmp18
    return (out1,)
end
