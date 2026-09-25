# module_key=1_5
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_5/L16_WD01_asinh_R18_512__4/checkpoint_0.pickle
# best_subproblem_loss=0.7107778765011563

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(reduce_smallerAxis, x1; return_type=Float64)
    tmp2 = safe_call(glcm_glcm_entropy_std, x1; return_type=Float64)
    tmp3 = safe_call(morpholaplace_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(experimental_standardize_2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(orientation_select, tmp4; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(meanpool, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(moffat25_image2D, tmp6, tmp2, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(dominant_orientation, tmp7; return_type=Float64)
    tmp9 = safe_call(identity_float, tmp8; return_type=Float64)
    out1 = tmp9
    return (out1,)
end
