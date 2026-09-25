# module_key=1_2
# checkpoint_path=ORGANCMNIST/L16_WD01_asinh_R18_128/1_2/L16_WD01_asinh_R18_128__7/checkpoint_0.pickle
# best_subproblem_loss=0.11151233883106493

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(maxpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(iqrpool, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(maxpool_blocks, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(binarize_polysegment2D, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp5 = safe_call(subtract_img2D, tmp4, tmp3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(grad_orientation, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(findlocalminima_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(reduce_propBlack, tmp7; return_type=Float64)
    tmp9 = safe_call(zeros_2D, tmp7; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(fastscanning_image2D, tmp9; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp11 = safe_call(glcm_glcm_var_ref_std, tmp10; return_type=Float64)
    tmp12 = safe_call(binarize_sauvola2D, tmp2, tmp11, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(opening_2D, tmp12; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp14 = safe_call(subtract_img2D, tmp13, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(reduce_propBlack, tmp14; return_type=Float64)
    tmp16 = safe_call(orientation_energy_135, x1; return_type=Float64)
    tmp17 = safe_call(number_minus, tmp16, tmp15; return_type=Float64)
    tmp18 = safe_call(identity_float, tmp17; return_type=Float64)
    out1 = tmp18
    return (out1,)
end
