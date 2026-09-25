# module_key=1_3
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_3/L16_WD01_asinh_R18_128__103/checkpoint_0.pickle
# best_subproblem_loss=0.1872527214596349

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(glcm_max_prob_std, x1; return_type=Float64)
    tmp2 = safe_call(grad_orientation, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(iqrpool, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(orientation_spread, tmp3; return_type=Float64)
    tmp5 = safe_call(glcm_max_prob_mean, x1; return_type=Float64)
    tmp6 = safe_call(meanpool, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(gaussian5_image2D, tmp6, tmp5; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(region_sum_10p, tmp7, tmp4, tmp1; return_type=Float64)
    tmp9 = safe_call(medianpool, x1, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(experimental_tobinary_image2D, tmp9; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(reduce_propBlack, tmp10; return_type=Float64)
    tmp12 = safe_call(identity_float, tmp11; return_type=Float64)
    out1 = tmp12
    return (out1,)
end
