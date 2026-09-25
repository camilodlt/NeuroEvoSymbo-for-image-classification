# module_key=1_15
# checkpoint_path=OCTMNIST/L16_WD01_asinh_R18_512/1_15/L16_WD01_asinh_R18_512__2/checkpoint_0.pickle
# best_subproblem_loss=0.6366467468555335

# output_idx=1
function sequential_program(x1)
    tmp1 = safe_call(log_image2D, x1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_biggestAxis, tmp1; return_type=Float64)
    tmp3 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp4 = safe_call(zeros_2D, tmp3; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp5 = safe_call(glcm_glcm_mean_ref_sum, tmp4, tmp2; return_type=Float64)
    tmp6 = safe_call(glcm_IDM_std, x1; return_type=Float64)
    tmp7 = safe_call(gaussian13_image2D, tmp1; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp8 = safe_call(laplacian3_image2D, tmp7; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp9 = safe_call(ando5x_image2D, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp10 = safe_call(binarize_otsu2D, tmp9, tmp6; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp11 = safe_call(tophat_2D, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(glcm_glcm_entropy_sum, tmp11, tmp2, tmp5; return_type=Float64)
    tmp13 = safe_call(identity_float, tmp12; return_type=Float64)
    out1 = tmp13
    return (out1,)
end
