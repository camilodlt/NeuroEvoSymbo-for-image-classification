## Library 1

The functions in this library always return images of the same size in which pixels are values between 0 and 1.

| Fn | Description |
| --- | --- |
| identity_image2D | Returns the input image unchanged. |
| ones_2D | Creates an image filled with ones, matching input shape. |
| zeros_2D | Creates an image filled with zeros, matching input shape. |
| experimental_invert_2D | Inverts image intensities with respect to the [0, 1] range. |
| experimental_normalize_2D | Normalizes image values to the [0, 1] range. |
| experimental_standardize_2D | Standardizes image values using mean and standard deviation. |
| experimental_tointensity_image2D | Converts binary or segment images to intensity image representation. |
| erosion_2D | Applies morphological erosion to shrink bright regions. |
| dilation_2D | Applies morphological dilation to expand bright regions. |
| opening_2D | Applies opening (erosion followed by dilation) to remove small bright noise. |
| closing_2D | Applies closing (dilation followed by erosion) to fill small dark gaps. |
| tophat_2D | Computes top-hat transform to extract small bright structures. |
| bothat_2D | Computes (black) tophat transform to extract small dark structures. |
| morphogradient_2D | Computes morphological gradient to emphasize object boundaries. |
| morpholaplace_2D | Computes a morphological Laplace-style edge response. |
| subtract_img2D | Subtracts two images elementwise. |
| add_img2D | Adds two images elementwise. |
| mult_img2D | Multiplies two images elementwise. |
| max_img2D | Computes elementwise maximum between two images. |
| min_img2D | Computes elementwise minimum between two images. |
| bsubtract_image2D | Subtracts a scalar from every pixel of the image. |
| badd_image2D | Adds a scalar to every pixel of the image. |
| bmult_image2D | Multiplies every pixel of the image by a scalar. |
| sobelx_image2D | Applies Sobel X derivative filter. |
| sobely_image2D | Applies Sobel Y derivative filter. |
| sobelm_image2D | Computes Sobel gradient magnitude response. |
| ando3x_image2D | Applies Ando 3x3 X derivative filter. |
| ando3y_image2D | Applies Ando 3x3 Y derivative filter. |
| ando3m_image2D | Computes Ando 3x3 gradient magnitude response. |
| ando4x_image2D | Applies Ando 4x4 X derivative filter. |
| ando4y_image2D | Applies Ando 4x4 Y derivative filter. |
| ando4m_image2D | Computes Ando 4x4 gradient magnitude response. |
| ando5x_image2D | Applies Ando 5x5 X derivative filter. |
| ando5y_image2D | Applies Ando 5x5 Y derivative filter. |
| ando5m_image2D | Computes Ando 5x5 gradient magnitude response. |
| bickleyx_image2D | Applies Bickley X derivative filter. |
| bickleyy_image2D | Applies Bickley Y derivative filter. |
| bickleym_image2D | Computes Bickley gradient magnitude response. |
| prewittx_image2D | Applies Prewitt X derivative filter. |
| prewitty_image2D | Applies Prewitt Y derivative filter. |
| prewittm_image2D | Computes Prewitt gradient magnitude response. |
| scharrx_image2D | Applies Scharr X derivative filter. |
| scharry_image2D | Applies Scharr Y derivative filter. |
| scharrm_image2D | Computes Scharr gradient magnitude response. |
| gaussian5_image2D | Applies Gaussian smoothing with sigma 1 : 5x5. |
| gaussian9_image2D | Applies Gaussian smoothing with sigma 2 : 9x9. |
| gaussian13_image2D | Applies Gaussian smoothing with sigma 3 : 13x13. |
| gaussian17_image2D | Applies Gaussian smoothing with sigma 4 : 17x17. |
| gaussian25_image2D | Applies Gaussian smoothing with sigma 6 : 25x25. |
| laplacian3_image2D | Applies Laplacian filter for second-order edge response. |
| dog_image2D | Applies Difference-of-Gaussians filtering. |
| moffat5_image2D | Applies Moffat smoothing filter with kernel size 5x5. |
| moffat13_image2D | Applies Moffat smoothing filter with kernel size 13x13. |
| moffat25_image2D | Applies Moffat smoothing filter with kernel size 25x25. |
| exp_image2D | Applies exponential transform to each pixel. |
| loginv_image2D | Applies logarithm to intensity pixels and inverses sign. |
| log_image2D | Applies logarithm to intensity pixels and normalizes. |
| powerof_image2D | Raises each pixel value to a provided exponent. |
| if_else_multiplexer | Selects between two same-type values based on whether cond is greater than zero. |
| avgpool_blocks | Pools each non-overlapping block by mean and broadcasts that value within the block. |
| avgpool_cross_blocks | Pools each non-overlapping block by averaging its center row and column. |
| maxpool_blocks | Pools each non-overlapping block by maximum and broadcasts that value within the block. |
| maxpool_cross_blocks | Pools each non-overlapping block by max over its center row and column. |
| minpool_blocks | Pools each non-overlapping block by minimum and broadcasts that value within the block. |
| minpool_cross_blocks | Pools each non-overlapping block by min over its center row and column. |
| meanpool | Applies sliding-window mean pooling and resizes back to the original image size. |
| maxpool | Applies sliding-window max pooling and resizes back to the original image size. |
| minpool | Applies sliding-window min pooling and resizes back to the original image size. |
| stdpool | Applies sliding-window standard-deviation pooling and resizes back to the original image size. |
| medianpool | Applies sliding-window median pooling and resizes back to the original image size. |
| uniquecountpool | Applies sliding-window unique-count pooling and rescales values to [0, 1]. |
| argmaxcountpool | Applies sliding-window argmax-count pooling and rescales values to [0, 1]. |
| argmincountpool | Applies sliding-window argmin-count pooling and rescales values to [0, 1]. |
| iqrpool | Applies sliding-window interquartile-range pooling and resizes back to the original image size. |
| grad_magnitude | Computes gradient magnitude map of the input image. |
| grad_orientation | Computes gradient orientation map of the input image. |
| orientation_select | Subsets image gradients near a target orientation with configurable bandwidth. |

## Library 2

The functions in this library always return images of the same size in which pixels are booleans (0 or 1).

| Fn | Description |
| --- | --- |
| identity_image2D | Returns the input image unchanged. |
| ones_2D | Creates an image filled with ones, matching input shape. |
| zeros_2D | Creates an image filled with zeros, matching input shape. |
| experimental_invert_2D | Inverts binary image. |
| experimental_tobinary_image2D | Converts an intensity image to binary using default thresholding. |
| experimental_tobinary_th_image2D_factory | Converts an intensity image to binary using a provided threshold. |
| subtract_img2D | Subtracts two images elementwise and rounds. |
| add_img2D | Adds two images elementwise and rounds. |
| mult_img2D | Multiplies two images elementwise and rounds. |
| max_img2D | Computes elementwise maximum between two images and rounds. |
| min_img2D | Computes elementwise minimum between two images and rounds.|
| binarize_adaptive2D | Binarizes image with adaptive local thresholding. |
| binarize_niblack2D | Binarizes image using Niblack local thresholding. |
| binarize_polysegment2D | Binarizes image using polysegment thresholding. |
| binarize_sauvola2D | Binarizes image using Sauvola local thresholding. |
| binarize_otsu2D | Binarizes image using Otsu histogram threshold. |
| binarize_minimumintermodes2D | Binarizes image using minimum-intermodes histogram threshold. |
| binarize_intermodes2D | Binarizes image using intermodes histogram threshold. |
| binarize_minimumerror2D | Binarizes image using minimum-error threshold estimation. |
| binarize_moments2D | Binarizes image using moments-based histogram threshold. |
| binarize_unimodalrosin2D | Binarizes image using unimodal Rosin threshold. |
| binarize_entropy2D | Binarizes image using entropy-based thresholding. |
| binarize_balanced2D | Binarizes image using balanced histogram thresholding. |
| binarize_yen2D | Binarizes image using Yen thresholding method. |
| binarize_manual2D | Binarizes image using threshold parameters. |
| sobelx_image2D | Applies Sobel X derivative filter and rounds. |
| sobely_image2D | Applies Sobel Y derivative filter and rounds. |
| sobelm_image2D | Computes Sobel gradient magnitude response and rounds. |
| ando3x_image2D | Applies Ando 3x3 X derivative filter and rounds. |
| ando3y_image2D | Applies Ando 3x3 Y derivative filter and rounds. |
| ando3m_image2D | Computes Ando 3x3 gradient magnitude response and rounds. |
| ando4x_image2D | Applies Ando 4x4 X derivative filter and rounds. |
| ando4y_image2D | Applies Ando 4x4 Y derivative filter and rounds. |
| ando4m_image2D | Computes Ando 4x4 gradient magnitude response and rounds. |
| ando5x_image2D | Applies Ando 5x5 X derivative filter and rounds. |
| ando5y_image2D | Applies Ando 5x5 Y derivative filter and rounds. |
| ando5m_image2D | Computes Ando 5x5 gradient magnitude response and rounds. |
| bickleyx_image2D | Applies Bickley X derivative filter and rounds. |
| bickleyy_image2D | Applies Bickley Y derivative filter and rounds. |
| bickleym_image2D | Computes Bickley gradient magnitude response and rounds. |
| prewittx_image2D | Applies Prewitt X derivative filter and rounds. |
| prewitty_image2D | Applies Prewitt Y derivative filter and rounds. |
| prewittm_image2D | Computes Prewitt gradient magnitude response and rounds. |
| scharrx_image2D | Applies Scharr X derivative filter and rounds. |
| scharry_image2D | Applies Scharr Y derivative filter and rounds. |
| scharrm_image2D | Computes Scharr gradient magnitude response and rounds. |
| gaussian5_image2D | Applies Gaussian smoothing with sigma 1 : 5x5 and rounds. |
| gaussian9_image2D | Applies Gaussian smoothing with sigma 2 : 9x9 and rounds. |
| gaussian13_image2D | Applies Gaussian smoothing with sigma 3 : 13x13 and rounds. |
| gaussian17_image2D | Applies Gaussian smoothing with sigma 4 : 17x17 and rounds. |
| gaussian25_image2D | Applies Gaussian smoothing with sigma 6 : 25x25 and rounds. |
| laplacian3_image2D | Applies Laplacian filter for second-order edge response and rounds. |
| dog_image2D | Applies Difference-of-Gaussians filtering and rounds. |
| moffat5_image2D | Applies Moffat smoothing filter with kernel size 5x5 and rounds. |
| moffat13_image2D | Applies Moffat smoothing filter with kernel size 13x13 and rounds. |
| moffat25_image2D | Applies Moffat smoothing filter with kernel size 25x25 and rounds. |
| findlocalminima_image2D | Marks local minima locations in the image. |
| findlocalmaxima_image2D | Marks local maxima locations in the image. |
| erosion_2D | Applies morphological erosion to shrink bright regions. |
| dilation_2D | Applies morphological dilation to expand bright regions. |
| opening_2D | Applies opening (erosion followed by dilation) to remove small bright noise. |
| closing_2D | Applies closing (dilation followed by erosion) to fill small dark gaps. |
| tophat_2D | Computes top-hat transform to extract small bright structures. |
| bothat_2D | Computes (black) tophat transform to extract small dark structures. |
| morphogradient_2D | Computes morphological gradient to emphasize object boundaries. |
| morpholaplace_2D | Computes a morphological Laplace-style edge response. |
| if_else_multiplexer | Selects between two same-type values based on whether cond is greater than zero. |
| avgpool_blocks | Pools each non-overlapping block by mean and broadcasts that value within the block and rounds. |
| avgpool_cross_blocks | Pools each non-overlapping block by averaging its center row and column and rounds. |
| maxpool_blocks | Pools each non-overlapping block by maximum and broadcasts that value within the block and rounds. |
| maxpool_cross_blocks | Pools each non-overlapping block by max over its center row and column and rounds. |
| minpool_blocks | Pools each non-overlapping block by minimum and broadcasts that value within the block and rounds. |
| minpool_cross_blocks | Pools each non-overlapping block by min over its center row and column and rounds. |
| meanpool | Applies sliding-window mean pooling and resizes back to the original image size and rounds. |
| maxpool | Applies sliding-window max pooling and resizes back to the original image size and rounds. |
| minpool | Applies sliding-window min pooling and resizes back to the original image size and rounds. |
| stdpool | Applies sliding-window standard-deviation pooling and resizes back to the original image size and rounds. |
| medianpool | Applies sliding-window median pooling and resizes back to the original image size and rounds. |
| uniquecountpool | Applies sliding-window unique-count pooling and rescales values to [0, 1] and rounds. |
| argmaxcountpool | Applies sliding-window argmax-count pooling and rescales values to [0, 1] and rounds. |
| argmincountpool | Applies sliding-window argmin-count pooling and rescales values to [0, 1] and rounds. |
| iqrpool | Applies sliding-window interquartile-range pooling and resizes back to the original image size and rounds. |

## Library 3

The functions in this library always return images of the same size in which pixels are Integers (each integer correspond to a different segment).

| Fn | Description |
| --- | --- |
| identity_image2D | Returns the input image unchanged. |
| ones_2D | Creates an image filled with ones, matching input shape. |
| zeros_2D | Creates an image filled with zeros, matching input shape. |
| experimental_tosegment_image2D | Converts a binary image to segment image representation (separates background and foreground). |
| fastscanning_image2D | Segments the image using a fast scanning region-labeling strategy. |
| watershed_image2D | Segments the image using watershed transform over intensity topology. |

## Library 4

The functions in this library always return a scalar value.

| Fn | Description |
| --- | --- |
| identity_float | Returns the input Float64 value unchanged. |
| ret_1 | Returns the constant float value 1.0. |
| tanh | Applies hyperbolic tangent to a numeric input. |
| relu | Applies ReLU by returning max(x, 0) for numeric inputs. |
| modulo | Computes the modulo remainder a % b for two numeric inputs. |
| is_eq_to | Returns 1 when two numeric inputs are equal, otherwise 0. |
| experimental_is_gt | Returns 1 when the first number is greater than the second, otherwise 0. |
| experimental_is_lt | Returns 1 when the first number is lower than the second, otherwise 0. |
| experimental_not | Returns 1 when the input is at most 0.5, otherwise 0. |
| number_sum | Adds two numeric inputs. |
| number_minus | Subtracts the second numeric input from the first. |
| number_mult | Multiplies two numeric inputs. |
| number_div | Divides the first numeric input by the second and throws on division by zero. |
| safe_div | Divides two numeric inputs and returns 0 when the divisor is zero. |
| power_of | Raises the first numeric input to the power of the second. |
| pi_ | Returns the mathematical constant pi as Float64. |
| exp_ | Computes the exponential of the numeric input. |
| log_ | Computes the natural logarithm after clipping the input to a positive minimum. |
| log10_ | Computes the base-10 logarithm after clipping the input to a positive minimum. |
| reduce_length | Returns the number of pixels in the input image. |
| reduce_biggestAxis | Returns the largest image dimension size. |
| reduce_smallerAxis | Returns the smallest image dimension size. |
| reduce_histMode | Returns the most frequent pixel value in the image. |
| reduce_histModeCount | Returns the count of the most frequent pixel value in the image. |
| reduce_propWhite | Returns the proportion of white pixels in a binary image. |
| reduce_propBlack | Returns the proportion of black pixels in a binary image. |
| reduce_nColors | Returns the number of unique pixel values in the image. |
| reduce_mean | Returns the mean intensity of the image. |
| reduce_median | Returns the median intensity of the image. |
| reduce_std | Returns the standard deviation of image intensities. |
| reduce_maximum | Returns the maximum intensity value in the image. |
| reduce_minimum | Returns the minimum intensity value in the image. |
| if_else_multiplexer | Selects between two same-type values based on whether cond is greater than zero. |
| region_mean | Computes local mean intensity inside a patch centered at normalized coordinates. |
| region_std | Computes local standard deviation inside a patch centered at normalized coordinates. |
| region_min | Computes the minimum local value inside a centered patch. |
| region_max | Computes the maximum local value inside a centered patch. |
| region_sum | Computes the local sum of values inside a centered patch. |
| region_median | Computes the local median value inside a centered patch. |
| region_range | Computes local range as max minus min inside a centered patch. |
| region_contrast | Computes local contrast between a center patch and its surrounding ring. |
| region_energy | Computes local energy as the mean squared value inside a centered patch. |
| region_entropy | Computes local entropy from a coarse histogram inside a centered patch. |
| region_mean_5p | Computes local mean intensity using a percentage-sized patch centered at normalized coordinates (5%). |
| region_std_5p | Computes local standard deviation using a percentage-sized patch centered at normalized coordinates (5%). |
| region_min_5p | Computes the local minimum using a percentage-sized patch centered at normalized coordinates (5%). |
| region_max_5p | Computes the local maximum using a percentage-sized patch centered at normalized coordinates (5%). |
| region_sum_5p | Computes local sum using a percentage-sized patch centered at normalized coordinates (5%). |
| region_median_5p | Computes local median using a percentage-sized patch centered at normalized coordinates (5%). |
| region_range_5p | Computes local range using a percentage-sized patch centered at normalized coordinates (5%). |
| region_contrast_5p | Computes center-versus-ring local contrast using a percentage-sized patch (5%). |
| region_energy_5p | Computes local energy using a percentage-sized patch centered at normalized coordinates (5%). |
| region_entropy_5p | Computes local entropy from histogram bins in a percentage-sized patch (5%). |
| region_mean_10p | Computes local mean intensity using a percentage-sized patch centered at normalized coordinates (10%). |
| region_std_10p | Computes local standard deviation using a percentage-sized patch centered at normalized coordinates (10%). |
| region_min_10p | Computes the local minimum using a percentage-sized patch centered at normalized coordinates (10%). |
| region_max_10p | Computes the local maximum using a percentage-sized patch centered at normalized coordinates (10%). |
| region_sum_10p | Computes local sum using a percentage-sized patch centered at normalized coordinates (10%). |
| region_median_10p | Computes local median using a percentage-sized patch centered at normalized coordinates (10%). |
| region_range_10p | Computes local range using a percentage-sized patch centered at normalized coordinates (10%). |
| region_contrast_10p | Computes center-versus-ring local contrast using a percentage-sized patch (10%). |
| region_energy_10p | Computes local energy using a percentage-sized patch centered at normalized coordinates (10%). |
| region_entropy_10p | Computes local entropy from histogram bins in a percentage-sized patch (10%). |
| region_mean_20p | Computes local mean intensity using a percentage-sized patch centered at normalized coordinates (20%). |
| region_std_20p | Computes local standard deviation using a percentage-sized patch centered at normalized coordinates (20%). |
| region_min_20p | Computes the local minimum using a percentage-sized patch centered at normalized coordinates (20%). |
| region_max_20p | Computes the local maximum using a percentage-sized patch centered at normalized coordinates (20%). |
| region_sum_20p | Computes local sum using a percentage-sized patch centered at normalized coordinates (20%). |
| region_median_20p | Computes local median using a percentage-sized patch centered at normalized coordinates (20%). |
| region_range_20p | Computes local range using a percentage-sized patch centered at normalized coordinates (20%). |
| region_contrast_20p | Computes center-versus-ring local contrast using a percentage-sized patch (20%). |
| region_energy_20p | Computes local energy using a percentage-sized patch centered at normalized coordinates (20%). |
| region_entropy_20p | Computes local entropy from histogram bins in a percentage-sized patch (20%). |
| haar_lr | Computes left-versus-right Haar contrast in a local image region. |
| haar_tb | Computes top-versus-bottom Haar contrast in a local image region. |
| haar_diag_main | Computes diagonal Haar contrast favoring main-diagonal quadrants. |
| haar_diag_anti | Computes diagonal Haar contrast favoring anti-diagonal quadrants. |
| haar_center_surround | Computes center-versus-surround Haar contrast in a local region. |
| haar_three_h | Computes horizontal three-band Haar contrast in a local region. |
| haar_three_v | Computes vertical three-band Haar contrast in a local region. |
| orientation_coherence | Computes orientation coherence of the image gradient field. |
| dominant_orientation | Returns the strongest coarse orientation bin center among {0, π/4, π/2, 3π/4}, normalized by π. |
| orientation_energy_0 | Returns orientation energy proportion near 0 degrees. |
| orientation_energy_45 | Returns orientation energy proportion near 45 degrees. |
| orientation_energy_90 | Returns orientation energy proportion near 90 degrees. |
| orientation_energy_135 | Returns orientation energy proportion near 135 degrees. |
| orientation_spread | Measures how widely orientation energy is distributed across directions. |
| glcm_glcm_mean_ref_mean | Computes glcm mean metric across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_glcm_mean_ref_sum | Computes glcm mean metric across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_glcm_mean_ref_std | Computes glcm mean metric across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_glcm_mean_ref_minimum | Computes glcm mean metric across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_glcm_mean_ref_maximum | Computes glcm mean metric across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_glcm_var_ref_mean | Computes glcm var metric across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_glcm_var_ref_sum | Computes glcm var metric across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_glcm_var_ref_std | Computes glcm var metric across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_glcm_var_ref_minimum | Computes glcm var metric across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_glcm_var_ref_maximum | Computes glcm var metric across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_correlation_mean | Computes GLCM correlation across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_correlation_sum | Computes GLCM correlation across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_correlation_std | Computes GLCM correlation across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_correlation_minimum | Computes GLCM correlation across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_correlation_maximum | Computes GLCM correlation across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_contrast_mean | Computes GLCM contrast across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_contrast_sum | Computes GLCM contrast across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_contrast_std | Computes GLCM contrast across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_contrast_minimum | Computes GLCM contrast across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_contrast_maximum | Computes GLCM contrast across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_IDM_mean | Computes GLCM inverse-difference-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_IDM_sum | Computes GLCM inverse-difference-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_IDM_std | Computes GLCM inverse-difference-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer.|
| glcm_IDM_minimum | Computes GLCM inverse-difference-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_IDM_maximum | Computes GLCM inverse-difference-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_ASM_mean | Computes GLCM angular-second-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_ASM_sum | Computes GLCM angular-second-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_ASM_std | Computes GLCM angular-second-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_ASM_minimum | Computes GLCM angular-second-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_ASM_maximum | Computes GLCM angular-second-moment across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_glcm_entropy_mean | Computes GLCM entropy across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_glcm_entropy_sum | Computes GLCM entropy across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_glcm_entropy_std | Computes GLCM entropy across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_glcm_entropy_minimum | Computes GLCM entropy across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_glcm_entropy_maximum | Computes GLCM entropy across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_max_prob_mean | Computes GLCM max-probability across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_max_prob_sum | Computes GLCM max-probability across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_max_prob_std | Computes GLCM max-probability across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_max_prob_minimum | Computes GLCM max-probability across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_max_prob_maximum | Computes GLCM max-probability across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_energy_mean | Computes GLCM energy across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_energy_sum | Computes GLCM energy across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_energy_std | Computes GLCM energy across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_energy_minimum | Computes GLCM energy across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_energy_maximum | Computes GLCM energy across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
| glcm_dissimilarity_mean | Computes GLCM dissimilarity across angles [pi, 3pi/4, pi/2, pi/4], then applies mean reducer. |
| glcm_dissimilarity_sum | Computes GLCM dissimilarity across angles [pi, 3pi/4, pi/2, pi/4], then applies sum reducer. |
| glcm_dissimilarity_std | Computes GLCM dissimilarity across angles [pi, 3pi/4, pi/2, pi/4], then applies standard deviation reducer. |
| glcm_dissimilarity_minimum | Computes GLCM dissimilarity across angles [pi, 3pi/4, pi/2, pi/4], then applies minimum reducer. |
| glcm_dissimilarity_maximum | Computes GLCM dissimilarity across angles [pi, 3pi/4, pi/2, pi/4], then applies maximum reducer. |
