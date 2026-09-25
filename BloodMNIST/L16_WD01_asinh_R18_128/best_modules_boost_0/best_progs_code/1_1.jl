# module_key=1_1
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_1/L16_WD01_asinh_R18_128__4/checkpoint_0.pickle
# best_subproblem_loss=0.2791748487797564

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_energy_maximum, x3; return_type=Float64)
    tmp2 = safe_call(glcm_energy_std, x3, tmp1; return_type=Float64)
    tmp3 = safe_call(morphogradient_2D, x1, tmp1, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(binarize_yen2D, tmp3, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(dilation_2D, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(reduce_smallerAxis, x1; return_type=Float64)
    tmp7 = safe_call(gaussian5_image2D, x2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(iqrpool, tmp7, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(watershed_image2D, tmp8; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_glcm_mean_ref_sum, tmp9; return_type=Float64)
    tmp11 = safe_call(min_img2D, x3, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(maxpool, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(binarize_balanced2D, tmp12, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(add_img2D, tmp13, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(reduce_propBlack, tmp14; return_type=Float64)
    tmp16 = safe_call(identity_float, tmp15; return_type=Float64)
    out1 = tmp16
    return (out1,)
end
