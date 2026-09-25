# module_key=1_10
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_10/L16_WD01_asinh_R18_128__6/checkpoint_0.pickle
# best_subproblem_loss=0.18552781434387422

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(argmaxcountpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(dog_image2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(opening_2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(glcm_ASM_mean, tmp3; return_type=Float64)
    tmp5 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(reduce_nColors, tmp5; return_type=Float64)
    tmp7 = safe_call(reduce_median, x1; return_type=Float64)
    tmp8 = safe_call(binarize_manual2D, x1, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(medianpool, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(region_max_10p, tmp9, tmp6, tmp4; return_type=Float64)
    tmp11 = safe_call(pi_, ; return_type=Float64)
    tmp12 = safe_call(gaussian5_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(ret_1, ; return_type=Float64)
    tmp14 = safe_call(prewittx_image2D, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(laplacian3_image2D, tmp14, tmp13; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(max_img2D, tmp15, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(min_img2D, tmp16, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(morphogradient_2D, tmp17, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(ando5m_image2D, tmp3, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(subtract_img2D, x1, tmp19; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp21 = safe_call(max_img2D, tmp20, tmp18; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp22 = safe_call(glcm_energy_std, tmp21, tmp10; return_type=Float64)
    tmp23 = safe_call(glcm_IDM_std, tmp1, tmp7; return_type=Float64)
    tmp24 = safe_call(number_minus, tmp23, tmp22; return_type=Float64)
    tmp25 = safe_call(identity_float, tmp24; return_type=Float64)
    out1 = tmp25
    return (out1,)
end
