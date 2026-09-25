# module_key=1_14
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_14/L16_WD01_asinh_R18_512__101/checkpoint_0.pickle
# best_subproblem_loss=0.44963141894600345

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(reduce_smallerAxis, x1; return_type=Float64)
    tmp2 = safe_call(binarize_niblack2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(minpool, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(glcm_contrast_std, x1; return_type=Float64)
    tmp5 = safe_call(ando3x_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(prewittm_image2D, tmp5, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(mult_img2D, tmp6, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(gaussian5_image2D, tmp7, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(orientation_spread, tmp8; return_type=Float64)
    tmp10 = safe_call(glcm_correlation_minimum, x1; return_type=Float64)
    tmp11 = safe_call(erosion_2D, ; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp12 = safe_call(prewitty_image2D, tmp11, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(reduce_histMode, tmp12; return_type=Float64)
    tmp14 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp15 = safe_call(glcm_correlation_mean, tmp14; return_type=Float64)
    tmp16 = safe_call(region_entropy, x1, tmp10, tmp4; return_type=Float64)
    tmp17 = safe_call(moffat13_image2D, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(moffat25_image2D, tmp17; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(scharrx_image2D, tmp18; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(ando3m_image2D, tmp19, tmp16; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp21 = safe_call(prewittx_image2D, tmp20, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp22 = safe_call(glcm_dissimilarity_minimum, tmp21, tmp13; return_type=Float64)
    tmp23 = safe_call(power_of, tmp22, tmp9; return_type=Float64)
    tmp24 = safe_call(identity_float, tmp23; return_type=Float64)
    out1 = tmp24
    return (out1,)
end
