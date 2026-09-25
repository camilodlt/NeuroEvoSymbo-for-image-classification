# module_key=1_3
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_3/L16_WD01_asinh_R18_512__1/checkpoint_0.pickle
# best_subproblem_loss=0.30468914913486134

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(watershed_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp2 = safe_call(glcm_glcm_entropy_maximum, tmp1; return_type=Float64)
    tmp3 = safe_call(bothat_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(ando4m_image2D, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(gaussian5_image2D, tmp4, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(reduce_biggestAxis, x1; return_type=Float64)
    tmp7 = safe_call(glcm_IDM_mean, x1, tmp6, tmp2; return_type=Float64)
    tmp8 = safe_call(ando5m_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(bothat_2D, tmp8, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(mult_img2D, tmp9, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(orientation_spread, tmp10; return_type=Float64)
    tmp12 = safe_call(identity_float, tmp11; return_type=Float64)
    out1 = tmp12
    return (out1,)
end
