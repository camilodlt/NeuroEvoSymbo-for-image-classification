# module_key=1_6
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_6/L16_WD01_asinh_R18_128__102/checkpoint_0.pickle
# best_subproblem_loss=0.23619599438637484

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(zeros_2D, tmp1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp3 = safe_call(glcm_IDM_sum, tmp2; return_type=Float64)
    tmp4 = safe_call(glcm_correlation_maximum, x1; return_type=Float64)
    tmp5 = safe_call(binarize_unimodalrosin2D, ; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(gaussian13_image2D, tmp5, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp7 = safe_call(haar_center_surround, tmp6, tmp4, tmp3; return_type=Float64)
    tmp8 = safe_call(glcm_energy_maximum, x1; return_type=Float64)
    tmp9 = safe_call(ret_1, ; return_type=Float64)
    tmp10 = safe_call(gaussian9_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(region_sum_20p, tmp10, tmp9, tmp8; return_type=Float64)
    tmp12 = safe_call(prewitty_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(medianpool, tmp12, tmp11, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(scharrm_image2D, tmp13, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(glcm_glcm_mean_ref_std, tmp14; return_type=Float64)
    tmp16 = safe_call(glcm_energy_std, x1; return_type=Float64)
    tmp17 = safe_call(glcm_energy_std, x1, tmp16; return_type=Float64)
    tmp18 = safe_call(laplacian3_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(prewitty_image2D, tmp18; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(moffat13_image2D, tmp19, tmp17, tmp16; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp21 = safe_call(glcm_energy_std, tmp20, tmp15; return_type=Float64)
    tmp22 = safe_call(log10_, tmp21; return_type=Float64)
    tmp23 = safe_call(identity_float, tmp22; return_type=Float64)
    out1 = tmp23
    return (out1,)
end
