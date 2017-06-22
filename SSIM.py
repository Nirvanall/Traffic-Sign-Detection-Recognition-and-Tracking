import numpy as np
from skimage.util.dtype import dtype_range
from scipy.ndimage import gaussian_filter
from skimage.util.arraycrop import crop

#imput images should be grayscale images
def structual_similarity_ssim(imga, imgb):
    K1 = 0.01      # small constant parameter in ssim algorithm
    K2 = 0.03      # small constant parameter in ssim algorithm
    sigma = 1.5    # each patch mean and variance spatially weighted by a normalized Gaussian kernel of
                   # width sigma =1.5
    window = 11    # the sliding window size


    datamin, datamax = dtype_range[imga.dtype.type]    #TODO: make sure of the usage
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

    C1 = (K1 * datarange) ** 2    # TODO: give comments from paper
    C2 = (K2 * datarange) ** 2

    numerator = (2 * mux * muy + C1) * (2 * covxy + C2)
    denominator = (mux ** 2 + muy ** 2 + C1 ) * (variancex + variancey + C2)

    S = numerator / denominator    # range from -1 ~ 1, S = 1 means two image(i,j) are identical

    #pad = (window - 1) // 2

    meanS = S.mean()

    return S, meanS












