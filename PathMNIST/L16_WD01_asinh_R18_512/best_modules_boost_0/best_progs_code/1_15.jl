# module_key=1_15
# checkpoint_path=PathMNIST/L16_WD01_asinh_R18_512/1_15/L16_WD01_asinh_R18_512__103/checkpoint_0.pickle
# best_subproblem_loss=0.24491039572966777

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(glcm_glcm_var_ref_sum, x1; return_type=Float64)
    tmp2 = safe_call(glcm_IDM_mean, x1; return_type=Float64)
    tmp3 = safe_call(glcm_IDM_std, x2, tmp2; return_type=Float64)
    tmp4 = safe_call(experimental_not, tmp3; return_type=Float64)
    tmp5 = safe_call(glcm_ASM_mean, x3; return_type=Float64)
    tmp6 = safe_call(tophat_2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp7 = safe_call(mult_img2D, tmp6, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(bickleym_image2D, tmp7, tmp5; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(glcm_energy_std, tmp8, tmp4, tmp1; return_type=Float64)
    tmp10 = safe_call(binarize_niblack2D, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(watershed_image2D, tmp10, tmp10, tmp2; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp12 = safe_call(identity_image2D, tmp11; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp13 = safe_call(glcm_IDM_minimum, tmp12, tmp9; return_type=Float64)
    tmp14 = safe_call(erosion_2D, x1, tmp2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(experimental_invert_2D, tmp14; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp16 = safe_call(glcm_dissimilarity_sum, tmp15; return_type=Float64)
    tmp17 = safe_call(safe_div, tmp16, tmp13; return_type=Float64)
    tmp18 = safe_call(identity_float, tmp17; return_type=Float64)
    out1 = tmp18
    return (out1,)
end
