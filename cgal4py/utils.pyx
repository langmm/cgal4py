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
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be sorted along.

    Returns:
        np.ndarray of uint64: Indices that sort pos along dimension d.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef np.ndarray[np.uint64_t, ndim=1] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    quickSort(&pos[0,0], &idx[0], ndim, d, l, r)
    return idx

def py_insertSort(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d):
    r"""Get the indices required to sort coordinates along one dimension.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be sorted along.

    Returns:
        np.ndarray of uint64: Indices that sort pos along dimension d.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef np.ndarray[np.uint64_t, ndim=1] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    insertSort(&pos[0,0], &idx[0], ndim, d, l, r)
    return idx

def py_pivot(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d):
    r"""Get the index of the median of medians along one dimension and indices 
    that partition pos according to the median of medians.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be partitioned along.

    Returns:
        tuple of int64 and np.ndarray of uint64: Index q of idx that is the 
            median of medians & array of indices that partition pos such that 
            the first q elements are less than the median of medians.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef np.ndarray[np.uint64_t, ndim=1] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef int64_t q = pivot(&pos[0,0], &idx[0], ndim, d, l, r)
    return q+1, idx

def py_partition(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d,
                 np.int64_t p):
    r"""Get the indices required to partition coordinates along one dimension.

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates. 
        d (np.uint32_t): Dimension that pos should be partitioned along.
        p (np.int64_t): Element of pos[:,d] that pos should be used to partition 
            pos.

    Returns:
        tuple of int64 and np.ndarray of uint64: Number of elements q in pos 
            that are less that or equal to the pivot & array of indices that 
            partition pos such that the first q elements are less than the 
            pivot and the last (n-q) elements are greater than the pivot.

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef np.ndarray[np.uint64_t, ndim=1] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef int64_t q = partition(&pos[0,0], &idx[0], ndim, d, l, r, p)
    return q+1, idx

def py_select(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d,
              np.int64_t t):
    r"""Get the indices required to partition coordiantes such that the first 
    t elements in pos[:,d] are the smallest t elements in pos[:,d].

    Args:
        pos (np.ndarray of float64): (n,m) array of n m-D coordinates.
        d (np.uint32_t): Dimension that pos should be partitioned along. 
        t (np.int64_t): Number of smallest elements in pos[:,d] that should be 
            partitioned.

    Returns:
        np.ndarray of uint64: Indices required to partition pos such that the 
            1st t elements in pos[:,d] are the smallest t elements in pos[:,d].

    """
    cdef uint32_t ndim = pos.shape[1]
    cdef int64_t l = 0
    cdef int64_t r = pos.shape[0]-1
    cdef np.ndarray[np.uint64_t, ndim=1] idx
    idx = np.arange(pos.shape[0]).astype('uint64')
    cdef int64_t q = select(&pos[0,0], &idx[0], ndim, d, l, r, t)
    return idx


