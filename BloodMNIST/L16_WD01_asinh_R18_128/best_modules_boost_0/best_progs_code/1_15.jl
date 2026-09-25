# module_key=1_15
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_15/L16_WD01_asinh_R18_128__3/checkpoint_0.pickle
# best_subproblem_loss=0.28345095261187425

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(binarize_moments2D, x3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(reduce_propWhite, tmp1; return_type=Float64)
    tmp3 = safe_call(gaussian17_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(region_median_20p, tmp3, tmp2, tmp2; return_type=Float64)
    tmp5 = safe_call(dilation_2D, x2, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(glcm_max_prob_minimum, x3, tmp4; return_type=Float64)
    tmp7 = safe_call(binarize_entropy2D, x1, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(subtract_img2D, tmp7, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(reduce_mean, tmp8; return_type=Float64)
    tmp10 = safe_call(reduce_smallerAxis, x3; return_type=Float64)
    tmp11 = safe_call(experimental_is_gt, tmp4, tmp10; return_type=Float64)
    tmp12 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp13 = safe_call(region_energy, tmp12, tmp11, tmp2; return_type=Float64)
    tmp14 = safe_call(meanpool, tmp3, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(experimental_tobinary_th_image2D_factory, tmp14, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp16 = safe_call(region_sum_10p, tmp15, tmp13, tmp9; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
