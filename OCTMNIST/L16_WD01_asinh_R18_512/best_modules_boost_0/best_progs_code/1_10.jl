# module_key=1_10
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_10/L16_WD01_asinh_R18_512__101/checkpoint_0.pickle
# best_subproblem_loss=0.3935316565336284

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(reduce_minimum, x1; return_type=Float64)
    tmp2 = safe_call(fastscanning_image2D, x1; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp3 = safe_call(glcm_IDM_std, tmp2, tmp1, tmp1; return_type=Float64)
    tmp4 = safe_call(log_, tmp3; return_type=Float64)
    tmp5 = safe_call(laplacian3_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(gaussian25_image2D, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(orientation_energy_90, tmp6; return_type=Float64)
    tmp8 = safe_call(number_mult, tmp7, tmp4; return_type=Float64)
    tmp9 = safe_call(avgpool_cross_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(tophat_2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(badd_image2D, tmp10, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(orientation_coherence, tmp11; return_type=Float64)
    tmp13 = safe_call(identity_float, tmp12; return_type=Float64)
    out1 = tmp13
    return (out1,)
end
