
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
    cdef object pts_total
    # cdef np.ndarray[np.float64_t, ndim=2] pts_total

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
        self.pts_total = pts

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def consolidate_tess(self):
        from cgal4py.delaunay import _get_Delaunay
        cdef uint64_t ncells, ncells_out
        cdef info_t idx_inf = 0
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            ncells = <uint64_t>self.T.num_cells()
        cdef np.ndarray[np_info_t, ndim=2] allverts
        cdef np.ndarray[np_info_t, ndim=2] allneigh
        allverts = np.empty((ncells, self.T.ndim+1), np_info)
        allneigh = np.empty((ncells, self.T.ndim+1), np_info)
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            ncells_out = self.T.consolidate_tess(ncells, &idx_inf,
                                                 &allverts[0,0], &allneigh[0,0])
        allverts.resize(ncells_out, self.T.ndim+1, refcheck=False)
        allneigh.resize(ncells_out, self.T.ndim+1, refcheck=False)
        cdef np.ndarray[np_info_t, ndim=1] info_total
        info_total = np.empty(self.T.npts_total, np_info)
        cdef uint64_t i
        for i in range(self.T.npts_total):
            info_total[i] = self.T.idx_total[i]
        cdef object T = None
        if self.T.rank == 0:
            Delaunay = _get_Delaunay(self.T.ndim, bit64=(np_info==np.uint64))
            T = Delaunay()
            T.deserialize_with_info(self.pts_total, info_total,
                                    allverts, allneigh, idx_inf)
        return T
