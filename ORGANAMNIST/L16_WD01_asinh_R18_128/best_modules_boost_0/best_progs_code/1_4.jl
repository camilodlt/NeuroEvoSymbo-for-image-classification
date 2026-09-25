# module_key=1_4
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_4/L16_WD01_asinh_R18_128__1/checkpoint_0.pickle
# best_subproblem_loss=0.21693369544111896

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(fastscanning_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_energy_maximum, tmp1; return_type=Float64)
    tmp3 = safe_call(add_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(ando4x_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(bickleym_image2D, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(glcm_dissimilarity_sum, tmp5, tmp2, tmp2; return_type=Float64)
    tmp7 = safe_call(glcm_glcm_var_ref_maximum, tmp4, tmp6; return_type=Float64)
    tmp8 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp9 = safe_call(zeros_2D, tmp8; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp10 = safe_call(glcm_glcm_entropy_minimum, tmp9, tmp7; return_type=Float64)
    tmp11 = safe_call(glcm_IDM_maximum, tmp4; return_type=Float64)
    tmp12 = safe_call(loginv_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(bothat_2D, tmp12; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(ando3m_image2D, tmp13; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(ando3m_image2D, tmp14, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(glcm_energy_std, tmp15, tmp10; return_type=Float64)
    tmp17 = safe_call(erosion_2D, ; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(bickleym_image2D, tmp17; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(reduce_biggestAxis, tmp18; return_type=Float64)
    tmp20 = safe_call(glcm_glcm_var_ref_minimum, x1; return_type=Float64)
    tmp21 = safe_call(findlocalmaxima_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp22 = safe_call(laplacian3_image2D, tmp21, tmp20; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp23 = safe_call(region_median_20p, tmp22, tmp19, tmp19; return_type=Float64)
    tmp24 = safe_call(exp_, tmp23; return_type=Float64)
    tmp25 = safe_call(min_img2D, tmp4, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp26 = safe_call(grad_orientation, tmp25; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp27 = safe_call(gaussian13_image2D, tmp26, tmp19; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp28 = safe_call(glcm_correlation_sum, tmp27, tmp24, tmp16; return_type=Float64)
    tmp29 = safe_call(identity_float, tmp28; return_type=Float64)
    out1 = tmp29
    return (out1,)
end
