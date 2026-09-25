# module_key=1_8
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_8/L16_WD01_asinh_R18_512__101/checkpoint_0.pickle
# best_subproblem_loss=0.39014504616111667

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_contrast_sum, tmp1; return_type=Float64)
    tmp3 = safe_call(binarize_manual2D, x3, x2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(iqrpool, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(add_img2D, tmp4, x3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(identity_image2D, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(max_img2D, tmp6, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(glcm_dissimilarity_mean, tmp7, tmp2; return_type=Float64)
    tmp9 = safe_call(reduce_histModeCount, x2; return_type=Float64)
    tmp10 = safe_call(gaussian5_image2D, x1, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(dilation_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(min_img2D, tmp11, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(erosion_2D, tmp12, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(reduce_mean, tmp13; return_type=Float64)
    tmp15 = safe_call(identity_float, tmp14; return_type=Float64)
    out1 = tmp15
    return (out1,)
end
