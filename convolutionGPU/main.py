# import time
# start_time = time.time()
# import numba as nb
import numpy as np
from numba import cuda

from fft import fft_1d, ifft_1d
from transform import convert_to_complex

ix_g1 = np.fromfile("../data/IX.dat", dtype=np.complex128)
ix_g1 = ix_g1.astype(np.complex64)

ts_g1 = np.fromfile("../data/TS_G1.dat", dtype=np.int16)
ts_g1 = convert_to_complex(ts_g1)
ts_g1.shape = (-1, ix_g1.shape[0])


# @nb.njit(["complex64[:,:](complex64[:], complex64[:,:])"], cache=True)
@cuda.jit
def main_task(ix, ts):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    ts[pos] = ifft_1d(fft_1d(ts[pos]) * ix)

ix = np.fft.fft(ix_g1)
threadsperblock = 32
blockspergrid = (ts_g1.shape[0] + (threadsperblock - 1)) // threadsperblock
main_task[blockspergrid, threadsperblock](ix, ts_g1)



# print("numba time - ", time.time() - start_time)
# input("PRESS ENTER TO START")
#
# start_time = time.time()
# main_task(ix_g1, ts_g1)
# print("with compiling - ", time.time() - start_time)
#
# start_time = time.time()
# print(main_task(ix_g1, ts_g1))
# print("after compiling - ", time.time() - start_time)
