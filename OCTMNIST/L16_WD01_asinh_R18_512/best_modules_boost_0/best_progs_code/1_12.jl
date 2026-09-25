# module_key=1_12
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_12/L16_WD01_asinh_R18_512__1/checkpoint_0.pickle
# best_subproblem_loss=0.5409385684381773

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(orientation_spread, x1; return_type=Float64)
    tmp2 = safe_call(ando5m_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(gaussian9_image2D, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(bsubtract_image2D, tmp3, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(bickleym_image2D, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(argmincountpool, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_dissimilarity_maximum, tmp6; return_type=Float64)
    tmp8 = safe_call(erosion_2D, tmp3, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(subtract_img2D, tmp8, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(orientation_spread, tmp9; return_type=Float64)
    tmp11 = safe_call(identity_float, tmp10; return_type=Float64)
    out1 = tmp11
    return (out1,)
end
