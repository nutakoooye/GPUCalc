import time

import numpy as np
import numba as nb

from fft import fft_1d, ifft_1d
from transform import convert_to_complex

ix_g1 = np.fromfile("data/IX.dat", dtype=np.complex128)
ix_g1 = ix_g1.astype(np.complex64)

ts_g1 = np.fromfile("data/TS_G1.dat", dtype=np.int16)
ts_g1 = convert_to_complex(ts_g1)
ts_g1.shape = (-1, ix_g1.shape[0])


@nb.njit(["complex64[:,:](complex64[:], complex64[:,:])"], cache=True)
def main_task(ix, ts):
    convolution = np.empty_like(ts)
    ix = fft_1d(ix)
    for i in range(ts.shape[0]):
        spectral = fft_1d(ts[i])
        convolution[i] = ifft_1d(spectral * ix)

    return convolution


start_time = time.time()
main_task(ix_g1, ts_g1)
print("with compiling - ", time.time() - start_time)
start_time = time.time()
main_task(ix_g1, ts_g1)
print("after compiling - ", time.time() - start_time)
# ts_g1 = convert_to_complex(ts_g1)
# ts_g1.shape = (-1, ix_g1.shape[0])
# print(fft_1d(ts_g1[0]))
