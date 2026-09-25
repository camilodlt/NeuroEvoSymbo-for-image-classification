# module_key=1_6
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_6/L16_WD01_asinh_R18_512__104/checkpoint_0.pickle
# best_subproblem_loss=0.3780747610342863

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(glcm_ASM_maximum, x1; return_type=Float64)
    tmp2 = safe_call(binarize_manual2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(fastscanning_image2D, tmp2; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp4 = safe_call(glcm_contrast_maximum, tmp3; return_type=Float64)
    tmp5 = safe_call(reduce_std, x1; return_type=Float64)
    tmp6 = safe_call(log_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_glcm_mean_ref_sum, tmp6, tmp5; return_type=Float64)
    tmp8 = safe_call(glcm_max_prob_sum, x1; return_type=Float64)
    tmp9 = safe_call(gaussian17_image2D, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(laplacian3_image2D, tmp9, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(orientation_select, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(ando3y_image2D, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(tophat_2D, tmp12, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(ando3m_image2D, tmp13, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(region_mean_5p, tmp14, tmp4, tmp1; return_type=Float64)
    tmp16 = safe_call(ando5m_image2D, tmp11, tmp15; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(orientation_spread, tmp16; return_type=Float64)
    tmp18 = safe_call(avgpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(tophat_2D, tmp18; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(orientation_coherence, tmp19; return_type=Float64)
    tmp21 = safe_call(number_div, tmp20, tmp17; return_type=Float64)
    tmp22 = safe_call(identity_float, tmp21; return_type=Float64)
    out1 = tmp22
    return (out1,)
end
