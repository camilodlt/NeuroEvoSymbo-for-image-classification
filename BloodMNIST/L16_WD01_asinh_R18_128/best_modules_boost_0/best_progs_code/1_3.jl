# module_key=1_3
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_3/L16_WD01_asinh_R18_128__13/checkpoint_0.pickle
# best_subproblem_loss=0.11221581896514332

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(ando3m_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(binarize_otsu2D, x3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp3 = safe_call(min_img2D, tmp2, tmp1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp4 = safe_call(glcm_dissimilarity_maximum, x3; return_type=Float64)
    tmp5 = safe_call(sobelx_image2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp6 = safe_call(zeros_2D, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(glcm_glcm_var_ref_maximum, tmp6, tmp4; return_type=Float64)
    tmp8 = safe_call(minpool_blocks, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(gaussian17_image2D, tmp8, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(log_image2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(reduce_mean, x1; return_type=Float64)
    tmp12 = safe_call(gaussian17_image2D, x2, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(opening_2D, x3; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp14 = safe_call(gaussian5_image2D, tmp13, tmp11; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp15 = safe_call(max_img2D, tmp14, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(moffat5_image2D, tmp15; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp17 = safe_call(max_img2D, tmp16, tmp12; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp18 = safe_call(min_img2D, tmp17, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp19 = safe_call(add_img2D, tmp9, tmp18; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp20 = safe_call(erosion_2D, tmp19, tmp4; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp21 = safe_call(max_img2D, tmp20, tmp3; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp22 = safe_call(reduce_propBlack, tmp21; return_type=Float64)
    tmp23 = safe_call(identity_float, tmp22; return_type=Float64)
    out1 = tmp23
    return (out1,)
end
