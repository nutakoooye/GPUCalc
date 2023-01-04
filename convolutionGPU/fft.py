import cmath

import numpy as np
# import numba as nb
from numba import cuda


# @nb.jit(["intp(intp)"], cache=True)
@cuda.jit(device=True)
def ilog2(n):
    result = -1
    if n < 0:
        n = -n
    while n > 0:
        n >>= 1
        result += 1
    return result


# @nb.njit(["intp(intp, intp)"], fastmath=True, cache=True)
@cuda.jit(device=True)
def reverse_bits(val, width):
    result = 0
    for _ in range(width):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


# @nb.njit(["complex64[:](complex64[:])"], fastmath=True, cache=True)
@cuda.jit(device=True)
def fft_1d_radix2_rbi(arr):
    n = len(arr)
    levels = ilog2(n)
    e_arr = np.empty_like(arr)
    coeff = -2j * cmath.pi / n
    for i in range(n):
        e_arr[i] = cmath.exp(coeff * i)
    result = np.empty_like(arr)
    for i in range(n):
        result[i] = arr[reverse_bits(i, levels)]
    # Radix-2 decimation-in-time FFT
    size = 2
    while size <= n:
        half_size = size // 2
        step = n // size
        for i in range(0, n, size):
            k = 0
            for j in range(i, i + half_size):
                temp = result[j + half_size] * e_arr[k]
                result[j + half_size] = result[j] - temp
                result[j] += temp
                k += step
        size *= 2
    return result


# @nb.njit(["complex64[:](complex64[:])"], fastmath=True, cache=True)
@cuda.jit(device=True)
def fft_1d(arr):
    n = len(arr)
    if not n & (n - 1):
        return fft_1d_radix2_rbi(arr)
    else:
        raise RuntimeError("Array must have a length of a power of two")


# @nb.njit(["complex64[:](complex64[:])"], fastmath=True, cache=True)
@cuda.jit(device=True)
def ifft_1d(arr):
    arr_conjugate = np.conjugate(arr)
    fx = fft_1d(arr_conjugate)
    fx = np.conjugate(fx)
    fx = fx / arr.shape[0]

    return fx
