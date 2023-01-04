import numba as nb
import numpy as np


@nb.jit(["complex64[:](int16[:])"],cache=True)
def convert_to_complex(int_arr_g1):
    """
    converting an array of ints to an array of complex numbers
    [1, 2, 3, 4, 5, 6] -> [1.+2.j, 3.+4.j, 5.+6.j]

    :param ts_g1: ndarray[int16]
    :return: ndarray[complex64]
    """
    len_arr = int_arr_g1.shape[0]
    compl_arr_g1 = np.empty(len_arr // 2, dtype=np.complex64)
    for i in range(0, len_arr - 1, 2):
        compl_arr_g1[i//2] = complex(int_arr_g1[i], int_arr_g1[i + 1])
    return compl_arr_g1
