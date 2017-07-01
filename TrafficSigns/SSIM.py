import numpy as np
from skimage.util.dtype import dtype_range
from scipy.ndimage import gaussian_filter



def structual_similarity_ssim(imga, imgb):
    """
    Inputs: two comparison images.

    Parameters: all the parameters' values are identical with the reference paper.

    Output: S, meanS
        The full SSIM image matrix. The mean structural similarity of the S matrix.

    Note: imput images should be grayscale images.

    References: Wang, Z., Bovik, A. C., Sheikh, H. R. and Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity. IEEE
        Transactions on Image Processing, 13, 600-612.
    """

    K1 = 0.01      # small constant parameter in ssim algorithm
    K2 = 0.03      # small constant parameter in ssim algorithm
    sigma = 1.5    # each patch mean and variance spatially weighted by a normalized Gaussian kernel of
                   # width sigma =1.5
    window = 11    # the sliding window size


    datamin, datamax = dtype_range[imga.dtype.type]
    datarange = datamax - datamin

    imga = imga.astype(np.float64)
    imgb = imgb.astype(np.float64)

    norm_covariance = 1.0

    mux = gaussian_filter(imga, sigma)    # mean for image a
    muy = gaussian_filter(imgb, sigma)    # mean for imgae b

    mux2 = gaussian_filter(imga * imga, sigma)
    muy2 = gaussian_filter(imgb * imgb, sigma)
    muxy = gaussian_filter(imga * imgb, sigma)

    variancex = norm_covariance * (mux2 - mux * mux)    # variance for image a
    variancey = norm_covariance * (muy2 - muy * muy)    # variance for image b
    covxy = norm_covariance * (muxy - mux * muy)        # covariance for image a and b

    C1 = (K1 * datarange) ** 2
    C2 = (K2 * datarange) ** 2

    numerator = (2 * mux * muy + C1) * (2 * covxy + C2)
    denominator = (mux ** 2 + muy ** 2 + C1 ) * (variancex + variancey + C2)

    S = numerator / denominator    # elements range from -1 ~ 1, 1 means two image(i,j) are identical


    meanS = S.mean()

    return S, meanS
