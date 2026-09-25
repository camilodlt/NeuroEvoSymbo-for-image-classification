# module_key=1_7
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_7/L16_WD01_asinh_R18_512__2/checkpoint_0.pickle
# best_subproblem_loss=0.3385516472470391

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(ret_1, ; return_type=Float64)
    tmp2 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp3 = safe_call(glcm_dissimilarity_std, tmp2, tmp1; return_type=Float64)
    tmp4 = safe_call(avgpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(tophat_2D, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(orientation_coherence, tmp5; return_type=Float64)
    tmp7 = safe_call(fastscanning_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp8 = safe_call(zeros_2D, tmp7; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp9 = safe_call(glcm_glcm_mean_ref_sum, tmp8, tmp6, tmp3; return_type=Float64)
    tmp10 = safe_call(reduce_minimum, x1; return_type=Float64)
    tmp11 = safe_call(orientation_select, x1, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(binarize_sauvola2D, tmp11; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(erosion_2D, tmp12; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(gaussian25_image2D, tmp13, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(orientation_spread, tmp14; return_type=Float64)
    tmp16 = safe_call(number_minus, tmp6, tmp15; return_type=Float64)
    tmp17 = safe_call(identity_float, tmp16; return_type=Float64)
    out1 = tmp17
    return (out1,)
end
