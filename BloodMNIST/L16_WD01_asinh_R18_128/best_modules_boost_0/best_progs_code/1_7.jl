# module_key=1_7
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_7/L16_WD01_asinh_R18_128__2/checkpoint_0.pickle
# best_subproblem_loss=0.21409129295014728

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_energy_mean, x2; return_type=Float64)
    tmp2 = safe_call(minpool, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp3 = safe_call(subtract_img2D, tmp2, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp4 = safe_call(reduce_std, tmp3; return_type=Float64)
    tmp5 = safe_call(region_sum_10p, x1, tmp4, tmp1; return_type=Float64)
    tmp6 = safe_call(log_image2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(opening_2D, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp8 = safe_call(moffat25_image2D, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(mult_img2D, tmp8, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(iqrpool, tmp9, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(prewitty_image2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp12 = safe_call(ando4x_image2D, tmp11; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp13 = safe_call(dog_image2D, tmp12; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(binarize_otsu2D, tmp13, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(reduce_biggestAxis, tmp14; return_type=Float64)
    tmp16 = safe_call(bothat_2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(binarize_yen2D, tmp16; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(sobely_image2D, tmp17, tmp15; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp19 = safe_call(subtract_img2D, tmp18, tmp10; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(reduce_propBlack, tmp19; return_type=Float64)
    tmp21 = safe_call(identity_float, tmp20; return_type=Float64)
    out1 = tmp21
    return (out1,)
end
