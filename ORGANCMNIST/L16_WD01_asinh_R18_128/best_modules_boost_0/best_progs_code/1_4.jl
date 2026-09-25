# module_key=1_4
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_4/L16_WD01_asinh_R18_128__8/checkpoint_0.pickle
# best_subproblem_loss=0.22062921222150012

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(reduce_nColors, x1; return_type=Float64)
    tmp2 = safe_call(glcm_ASM_mean, x1; return_type=Float64)
    tmp3 = safe_call(mult_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(bickleym_image2D, tmp3, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(fastscanning_image2D, tmp4, tmp1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(reduce_nColors, tmp5; return_type=Float64)
    tmp7 = safe_call(minpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(glcm_correlation_std, tmp7; return_type=Float64)
    tmp9 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_energy_mean, tmp9, tmp8; return_type=Float64)
    tmp11 = safe_call(reduce_biggestAxis, x1; return_type=Float64)
    tmp12 = safe_call(sobelm_image2D, x1, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(ando3m_image2D, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(ando5x_image2D, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(glcm_max_prob_mean, tmp14, tmp11, tmp2; return_type=Float64)
    tmp16 = safe_call(uniquecountpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(ando3m_image2D, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(scharry_image2D, tmp17; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(mult_img2D, tmp18, tmp16; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(experimental_invert_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp21 = safe_call(subtract_img2D, tmp20, tmp19; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp22 = safe_call(powerof_image2D, tmp21, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp23 = safe_call(glcm_correlation_std, tmp22, tmp10, tmp6; return_type=Float64)
    tmp24 = safe_call(identity_float, tmp23; return_type=Float64)
    out1 = tmp24
    return (out1,)
end
