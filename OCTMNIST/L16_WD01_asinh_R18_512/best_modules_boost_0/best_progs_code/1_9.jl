# module_key=1_9
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_9/L16_WD01_asinh_R18_512__103/checkpoint_0.pickle
# best_subproblem_loss=0.4663224305560685

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(dilation_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(orientation_spread, tmp1; return_type=Float64)
    tmp3 = safe_call(tanh, tmp2; return_type=Float64)
    tmp4 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_max_prob_std, tmp4; return_type=Float64)
    tmp6 = safe_call(gaussian13_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_max_prob_std, tmp6; return_type=Float64)
    tmp8 = safe_call(laplacian3_image2D, tmp6; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(moffat5_image2D, tmp8, tmp7, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(orientation_energy_90, tmp9; return_type=Float64)
    tmp11 = safe_call(power_of, tmp10, tmp3; return_type=Float64)
    tmp12 = safe_call(identity_float, tmp11; return_type=Float64)
    out1 = tmp12
    return (out1,)
end
