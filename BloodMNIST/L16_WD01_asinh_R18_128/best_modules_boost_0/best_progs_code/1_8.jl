# module_key=1_8
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_8/L16_WD01_asinh_R18_128__13/checkpoint_0.pickle
# best_subproblem_loss=0.17887517685802556

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(findlocalminima_image2D, x2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(reduce_propBlack, tmp1; return_type=Float64)
    tmp3 = safe_call(glcm_ASM_std, x1; return_type=Float64)
    tmp4 = safe_call(glcm_IDM_minimum, x1, tmp3; return_type=Float64)
    tmp5 = safe_call(glcm_energy_std, x2; return_type=Float64)
    tmp6 = safe_call(moffat5_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(mult_img2D, tmp6, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(ando5m_image2D, tmp7, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(mult_img2D, tmp8, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(closing_2D, tmp9, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(reduce_propWhite, tmp10; return_type=Float64)
    tmp12 = safe_call(stdpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(reduce_nColors, tmp12; return_type=Float64)
    tmp14 = safe_call(fastscanning_image2D, tmp6; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp15 = safe_call(glcm_IDM_minimum, tmp14; return_type=Float64)
    tmp16 = safe_call(glcm_glcm_mean_ref_std, x2, tmp15; return_type=Float64)
    tmp17 = safe_call(binarize_entropy2D, x1, tmp16; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(avgpool_cross_blocks, tmp17, tmp13; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp19 = safe_call(region_sum_10p, tmp18, tmp11, tmp2; return_type=Float64)
    tmp20 = safe_call(identity_float, tmp19; return_type=Float64)
    out1 = tmp20
    return (out1,)
end
