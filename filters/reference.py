"""OpenCV reference implementation.

Used for two things: correctness (our outputs should match within 1 grey
level) and as a performance target ("how far are we from a tuned library?").
"""

import cv2
import numpy as np

from .kernel import default_sigma


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def blur_2d(gray, ksize, sigma=None):
    if sigma is None or sigma <= 0:
        sigma = default_sigma(ksize)
    return cv2.GaussianBlur(
        gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101
    )


def correlate_2d_float(gray, K):
    return cv2.filter2D(
        gray.astype(np.float32), cv2.CV_32F, np.asarray(K, np.float32),
        borderType=cv2.BORDER_REFLECT_101,
    )


def set_threads(n):
    cv2.setNumThreads(int(n))
