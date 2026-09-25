# module_key=1_13
# checkpoint_path=BloodMNIST/L16_WD01_asinh_R18_128/1_13/L16_WD01_asinh_R18_128__4/checkpoint_0.pickle
# best_subproblem_loss=0.18260920305467054

# output_idx=1
function sequential_program(x1, x2, x3)
    tmp1 = safe_call(avgpool_blocks, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp2 = safe_call(reduce_nColors, tmp1; return_type=Float64)
    tmp3 = safe_call(fastscanning_image2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp4 = safe_call(glcm_contrast_std, tmp3; return_type=Float64)
    tmp5 = safe_call(ones_2D, ; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp6 = safe_call(zeros_2D, tmp5; return_type=SImage2D{64, 64, SegmentPixel{Int64}, Matrix{SegmentPixel{Int64}}})
    tmp7 = safe_call(glcm_contrast_maximum, tmp6, tmp4; return_type=Float64)
    tmp8 = safe_call(mult_img2D, x1, x1; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp9 = safe_call(iqrpool, tmp8; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp10 = safe_call(prewittx_image2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp11 = safe_call(sobely_image2D, tmp10; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp12 = safe_call(experimental_standardize_2D, tmp11; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp13 = safe_call(region_std_10p, tmp12, tmp7, tmp2; return_type=Float64)
    tmp14 = safe_call(ando5y_image2D, tmp8; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp15 = safe_call(reduce_median, tmp14; return_type=Float64)
    tmp16 = safe_call(gaussian13_image2D, tmp9; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp17 = safe_call(region_entropy_20p, tmp16, tmp15, tmp15; return_type=Float64)
    tmp18 = safe_call(dominant_orientation, x2; return_type=Float64)
    tmp19 = safe_call(bothat_2D, x2; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp20 = safe_call(laplacian3_image2D, tmp19, tmp18; return_type=SImage2D{64, 64, IntensityPixel{N0f8}, Matrix{IntensityPixel{N0f8}}})
    tmp21 = safe_call(binarize_otsu2D, tmp20; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp22 = safe_call(minpool_blocks, tmp21; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp23 = safe_call(minpool, tmp22, tmp17, tmp13; return_type=SImage2D{64, 64, BinaryPixel{Bool}, Matrix{BinaryPixel{Bool}}})
    tmp24 = safe_call(reduce_propBlack, tmp23; return_type=Float64)
    tmp25 = safe_call(identity_float, tmp24; return_type=Float64)
    out1 = tmp25
    return (out1,)
end
