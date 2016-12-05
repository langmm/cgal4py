
import cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from cython.operator cimport dereference
from cython.operator cimport preincrement, predecrement
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t


cdef class ParallelDelaunay:

    cdef CParallelDelaunay *T

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T = new CParallelDelaunay()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run(self, np.ndarray[np.float64_t, ndim=2] pts,
            np.ndarray[np.float64_t, ndim=1] le,
            np.ndarray[np.float64_t, ndim=1] re,
            object periodic=False):
        cdef np.uint64_t npts = pts.shape[0]
        cdef np.uint32_t ndim = pts.shape[1]
        cdef cbool* per = <cbool *>malloc(ndim*sizeof(cbool));
        if isinstance(periodic, pybool):
            for i in range(ndim):
                per[i] = <cbool>periodic
        else:
            for i in range(ndim):
                per[i] = <cbool>periodic[i]
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.run(npts, ndim, &pts[0,0], &le[0], &re[0], per)
        
