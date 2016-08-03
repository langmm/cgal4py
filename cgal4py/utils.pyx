import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t

import copy

import scipy
from scipy.sparse import csr_matrix

def py_max_pts(np.ndarray[np.float64_t, ndim=2] pos):
    r"""Get the maximum of points along each coordinate. 

    Args: 
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 

    Returns: 
        np.ndarray of float64: Maximum of pos along each coordinate. 

    """
    cdef uint64_t n = <uint64_t>pos.shape[0]
    cdef uint32_t m = <uint32_t>pos.shape[1]
    cdef np.float64_t* cout = max_pts(&pos[0,0], n, m)
    cdef uint32_t i = 0
    cdef np.ndarray[np.float64_t] out = np.zeros(m, 'float64')
    for i in range(m):
        out[i] = cout[i]
    return out

def py_min_pts(np.ndarray[np.float64_t, ndim=2] pos):
    r"""Get the minimum of points along each coordinate. 

    Args: 
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 

    Returns: 
        np.ndarray of float64: Minimum of pos along each coordinate. 

    """
    cdef uint64_t n = <uint64_t>pos.shape[0]
    cdef uint32_t m = <uint32_t>pos.shape[1]
    cdef np.float64_t* cout = min_pts(&pos[0,0], n, m)
    cdef uint32_t i = 0
    cdef np.ndarray[np.float64_t] out = np.zeros(m, 'float64')
    for i in range(m):
        out[i] = cout[i]
    return out

def py_quickSort(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d):
    r"""Get the indices required to sort coordinates along one dimension.

    Args:

    Returns:

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef np.ndarray[np.uint64_t, ndim=1] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    quickSort(&pos[0,0], &idx[0], ndim, d, l, r)
    return idx

