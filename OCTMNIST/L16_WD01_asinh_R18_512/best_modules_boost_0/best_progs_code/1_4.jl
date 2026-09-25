# module_key=1_4
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_4/L16_WD01_asinh_R18_512__103/checkpoint_0.pickle
# best_subproblem_loss=0.24497811010145998

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(meanpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(experimental_normalize_2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(mult_img2D, tmp2, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(subtract_img2D, tmp1, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(mult_img2D, tmp4, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(orientation_spread, tmp5; return_type=Float64)
    tmp7 = safe_call(log10_, tmp6; return_type=Float64)
    tmp8 = safe_call(identity_float, tmp7; return_type=Float64)
    out1 = tmp8
    return (out1,)
end
