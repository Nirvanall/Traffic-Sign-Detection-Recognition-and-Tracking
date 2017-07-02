import cv2
import numpy as np


def color_histogram(rect):
    """compute a color histogram for an image patch"""
    b, g, r = cv2.split(rect)
    b.astype(float)
    g.astype(float)
    r.astype(float)
    hr, bin_e = np.histogram(r, bins=32)
    hg, bin_e = np.histogram(g, bins=32)
    hb, bin_e = np.histogram(b, bins=32)
    h = np.concatenate((hr, hg))
    h = np.concatenate((h, hb))
    h = (1.0 * h) / (1.0 * np.sum(h))
    return h


def weighted_choice(weights, n):
    """Return N random item indices with weighting defined by weights"""
    cs = np.cumsum(weights)
    indices = []
    for j in range(0, n):
        idx = sum(cs < np.random.rand())
        indices = np.hstack((indices, idx))
    return indices


def motion_model(s_t1, width, height):
    """implementation of sampling from a simple motion model"""
    s_t = s_t1
    sigma = 15
    n = 100  # number of particles
    for i in range(0, n):
        s_t[i][0] = np.math.sqrt(sigma) * np.random.randn() + s_t[i][0]
        s_t[i][1] = np.math.sqrt(sigma) * np.random.randn() + s_t[i][1]
    for j in range(0, n):
        s_t[j][0] = max(1, min(width - np.ceil(0.5 * s_t[j][4] * s_t[j][5]), s_t[j][0]))
        s_t[j][1] = max(1, min(height - np.ceil(0.5 * s_t[j][5]), s_t[j][1]))
    return s_t


def appearance_model(img, s_t, color_model, width, height):
    """implementation of a simple appearance model"""

    # extract the image patch corresponding to th current particles
    r = int(round(s_t[1]))
    c = int(round(s_t[0]))
    w = int(round((s_t[4] * s_t[5])))
    h = int(round(s_t[5]))
    r2 = min(height, (r + h + 1))
    c2 = min(width, (c + w + 1))
    patch = img[r:r2, c:c2]

    # compute the appearance likelihood z_t
    hist = color_histogram(patch)
    d = kl_divergence(color_model, hist)
    l = 10
    z_t = np.math.exp((-l * d))
    return z_t


def kl_divergence(h1n, h2n):
    """compute the KL divergence between two 1-D histograms"""

    eta = 0.00001
    h1n = [x+eta for x in h1n]
    h2n = [y+eta for y in h2n]
    temp = np.zeros(len(h1n), float)
    for l in range(0, len(h1n)):
        temp[l] = h1n[l] * np.math.log((h1n[l]/h2n[l]))
        if np.math.isnan(temp[l]):
            temp[l] = 0
    k = sum(temp)
    return k


def draw_box(img, estimate_t):
    r = int(round(estimate_t[0]))
    c = int(round(estimate_t[1]))
    w = int(round(estimate_t[4] * estimate_t[5]))
    h = int(round(estimate_t[5]))

    x1 = r
    x2 = r + w
    y1 = c
    y2 = c + h
    cv2.rectangle(img, (x1, y1), (x2, y2),  (255, 0, 0), 2)
