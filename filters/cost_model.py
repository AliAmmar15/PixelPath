"""FLOP and byte counts per output pixel, used for GFLOP/s, GiB/s and the Roofline.

Bytes are the *compulsory* traffic: what an ideal implementation must read
from and write to main memory, assuming perfect cache reuse of neighbouring
pixels. Real implementations (especially the NumPy stage) move more, which is
one reason they sit below the Roofline.

  grayscale       3 mul + 2 add = 5 FLOP   read 3 B (RGB) + write 1 B          -> I = 1.25
  blur, direct    k*k (mul + add) = 2k^2   read 1 B + write 1 B                -> I = k^2
  blur, separable 2 passes x 2k = 4k       1 read + 4 write tmp + 4 read tmp
                                           + 1 write (float32 temp) = 10 B     -> I = 0.4k
"""

GIB = 1024 ** 3


def flops_per_pixel(filter_name, variant, ksize=None):
    if filter_name == "grayscale":
        return 5
    if variant == "separable":
        return 4 * ksize
    return 2 * ksize * ksize


def bytes_per_pixel(filter_name, variant, ksize=None):
    if filter_name == "grayscale":
        return 4
    if variant == "separable":
        return 10
    return 2


def intensity(filter_name, variant, ksize=None):
    return flops_per_pixel(filter_name, variant, ksize) / bytes_per_pixel(filter_name, variant, ksize)
