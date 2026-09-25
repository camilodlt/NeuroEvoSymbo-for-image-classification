# module_key=1_11
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_11/L16_WD01_asinh_R18_512__104/checkpoint_0.pickle
# best_subproblem_loss=0.37806205947632165

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(glcm_glcm_mean_ref_sum, x1; return_type=Float64)
    tmp2 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp3 = safe_call(glcm_glcm_entropy_sum, tmp2; return_type=Float64)
    tmp4 = safe_call(fastscanning_image2D, x1, tmp1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_IDM_maximum, tmp4, tmp3; return_type=Float64)
    tmp6 = safe_call(glcm_glcm_entropy_maximum, x1; return_type=Float64)
    tmp7 = safe_call(tophat_2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(region_mean_10p, tmp7, tmp6, tmp5; return_type=Float64)
    tmp9 = safe_call(maxpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(uniquecountpool, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(moffat25_image2D, tmp10, tmp8, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(reduce_nColors, tmp10; return_type=Float64)
    tmp13 = safe_call(minpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(tophat_2D, tmp13, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(subtract_img2D, tmp14, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(orientation_coherence, tmp15; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
