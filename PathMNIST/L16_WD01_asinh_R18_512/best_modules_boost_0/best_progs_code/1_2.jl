# module_key=1_2
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_2/L16_WD01_asinh_R18_512__2/checkpoint_0.pickle
# best_subproblem_loss=0.3016105225062521

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(orientation_spread, x1; return_type=Float64)
    tmp2 = safe_call(reduce_std, x2; return_type=Float64)
    tmp3 = safe_call(gaussian9_image2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(log_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(powerof_image2D, tmp4, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(orientation_select, tmp5, tmp2, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(reduce_median, tmp6; return_type=Float64)
    tmp8 = safe_call(log10_, tmp7; return_type=Float64)
    tmp9 = safe_call(identity_float, tmp8; return_type=Float64)
    out1 = tmp9
    return (out1,)
end
