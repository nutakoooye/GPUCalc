import cmath

import numpy as np
import numba as nb


@nb.jit(["intp(intp)"], cache=True)
def ilog2(n):
    result = -1
    if n < 0:
        n = -n
    while n > 0:
        n >>= 1
        result += 1
    return result


@nb.njit(["intp(intp, intp)"], fastmath=True, cache=True)
def reverse_bits(val, width):
    result = 0
    for _ in range(width):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


@nb.njit(["complex64[:](complex64[:], boolean)"], fastmath=True, cache=True)
def fft_1d_radix2_rbi(arr, direct):
    n = len(arr)
    levels = ilog2(n)
    e_arr = np.empty_like(arr)
    coeff = (-2j if direct else 2j) * cmath.pi / n
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


@nb.njit(["complex64[:](complex64[:], complex64[:])"],
         fastmath=True,
         cache=True)
def fft_convolve(a_arr, b_arr):
    return fft_1d_radix2_rbi(
        fft_1d_radix2_rbi(a_arr, True)
        * fft_1d_radix2_rbi(b_arr, True),
        False)


@nb.njit(["complex64[:](complex64[:])"], fastmath=True, cache=True)
def fft_1d_arb(arr):
    n = len(arr)
    m = 1 << (ilog2(n) + 2)
    e_arr = np.empty(n, dtype=np.complex64)
    for i in range(n):
        e_arr[i] = cmath.exp(-1j * cmath.pi * (i * i) / n)
    result = np.zeros(m, dtype=np.complex64)
    result[:n] = arr * e_arr
    coeff = np.zeros_like(result)
    coeff[:n] = e_arr.conjugate()
    coeff[-n + 1:] = e_arr[:0:-1].conjugate()
    return fft_convolve(result, coeff)[:n] * e_arr / m


@nb.njit(["complex64[:](complex64[:])"], fastmath=True, cache=True)
def fft_1d(arr):
    n = len(arr)
    if not n & (n - 1):
        return fft_1d_radix2_rbi(arr, True)
    else:
        return fft_1d_arb(arr)


@nb.njit(["complex64[:](complex64[:])"], fastmath=True, cache=True)
def ifft_1d(fu):
    fu_conjugate = np.conjugate(fu)
    fx = fft_1d(fu_conjugate)
    fx = np.conjugate(fx)
    fx = fx / fu.shape[0]

    return fx
