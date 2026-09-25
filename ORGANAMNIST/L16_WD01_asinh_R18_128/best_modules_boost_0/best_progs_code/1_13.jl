# module_key=1_13
# checkpoint_path=ORGANAMNIST/L16_WD01_asinh_R18_128/1_13/L16_WD01_asinh_R18_128__1/checkpoint_0.pickle
# best_subproblem_loss=0.2705771718911758

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(opening_2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp2 = safe_call(experimental_invert_2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(gaussian17_image2D, tmp2; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(subtract_img2D, tmp3, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp5 = safe_call(scharry_image2D, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp6 = safe_call(reduce_propBlack, tmp5; return_type=Float64)
    tmp7 = safe_call(glcm_correlation_sum, x1; return_type=Float64)
    tmp8 = safe_call(tanh, tmp7; return_type=Float64)
    tmp9 = safe_call(gaussian25_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(moffat5_image2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(avgpool_blocks, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(region_range_5p, tmp11, tmp8, tmp7; return_type=Float64)
    tmp13 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp14 = safe_call(glcm_ASM_maximum, tmp13; return_type=Float64)
    tmp15 = safe_call(grad_orientation, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(gaussian5_image2D, tmp15, tmp14; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp17 = safe_call(mult_img2D, x1, tmp16; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp18 = safe_call(bickleym_image2D, tmp17, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(region_sum_20p, tmp18, tmp6, tmp6; return_type=Float64)
    tmp20 = safe_call(identity_float, tmp19; return_type=Float64)
    out1 = tmp20
    return (out1,)
end
