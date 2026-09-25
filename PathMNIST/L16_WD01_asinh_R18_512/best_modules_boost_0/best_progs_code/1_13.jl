# module_key=1_13
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_13/L16_WD01_asinh_R18_512__3/checkpoint_0.pickle
# best_subproblem_loss=0.4328692198025563

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_IDM_minimum, x2; return_type=Float64)
    tmp2 = safe_call(laplacian3_image2D, x1, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(prewittm_image2D, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(watershed_image2D, tmp3; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_IDM_sum, tmp4; return_type=Float64)
    tmp6 = safe_call(glcm_ASM_maximum, x2, tmp1; return_type=Float64)
    tmp7 = safe_call(bothat_2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(prewittm_image2D, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(max_img2D, tmp8, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(gaussian5_image2D, tmp9, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(erosion_2D, tmp10, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(binarize_moments2D, tmp11, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(reduce_propWhite, tmp12; return_type=Float64)
    tmp14 = safe_call(identity_float, tmp13; return_type=Float64)
    out1 = tmp14
    return (out1,)
end
