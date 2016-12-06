
import cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from cython.operator cimport dereference
from cython.operator cimport preincrement, predecrement
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t


ctypedef uint32_t info_t
cdef object np_info = np.uint32
ctypedef np.uint32_t np_info_t


cdef class ParallelDelaunay:

    cdef CParallelDelaunay[info_t] *T

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] le,
                  np.ndarray[np.float64_t, ndim=1] re,
                  object periodic=False):
        cdef np.uint32_t ndim = le.size
        cdef cbool* per = <cbool *>malloc(ndim*sizeof(cbool));
        if isinstance(periodic, pybool):
            for i in range(ndim):
                per[i] = <cbool>periodic
        else:
            for i in range(ndim):
                per[i] = <cbool>periodic[i]
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T = new CParallelDelaunay[info_t](ndim, &le[0], &re[0], per)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def insert(self, np.ndarray[np.float64_t, ndim=2] pts):
        cdef np.uint32_t ndim = pts.shape[1]
        assert(ndim == self.T.ndim)
        cdef np.uint64_t npts = pts.shape[0]
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.insert(npts, &pts[0,0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def consolidate_tess(self):
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.consolidate_tess()
