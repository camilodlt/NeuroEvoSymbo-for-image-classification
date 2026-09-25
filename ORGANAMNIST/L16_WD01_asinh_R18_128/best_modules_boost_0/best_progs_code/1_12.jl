# module_key=1_12
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_12/L16_WD01_asinh_R18_128__103/checkpoint_0.pickle
# best_subproblem_loss=0.3203517817515329

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(orientation_energy_0, x1; return_type=Float64)
    tmp2 = safe_call(loginv_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(loginv_image2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(mult_img2D, tmp3, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(sobelx_image2D, tmp4, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(prewittm_image2D, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(reduce_propBlack, tmp6; return_type=Float64)
    tmp8 = safe_call(glcm_correlation_mean, x1; return_type=Float64)
    tmp9 = safe_call(prewitty_image2D, x1, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(orientation_coherence, tmp9; return_type=Float64)
    tmp11 = safe_call(reduce_histMode, tmp3; return_type=Float64)
    tmp12 = safe_call(reduce_maximum, x1; return_type=Float64)
    tmp13 = safe_call(maxpool, tmp2, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(glcm_glcm_entropy_std, tmp13, tmp11; return_type=Float64)
    tmp15 = safe_call(binarize_adaptive2D, x1, tmp14; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp16 = safe_call(ando4m_image2D, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(gaussian17_image2D, tmp16; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(watershed_image2D, tmp5, tmp17, tmp10; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp19 = safe_call(region_mean_20p, tmp18, tmp10, tmp7; return_type=Float64)
    tmp20 = safe_call(identity_float, tmp19; return_type=Float64)
    out1 = tmp20
    return (out1,)
end
