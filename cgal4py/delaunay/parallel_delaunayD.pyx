# distutils: language = c++
import cython
import numpy as np
cimport numpy as np
from mpi4py import MPI
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from cython.operator cimport dereference
from cython.operator cimport preincrement, predecrement
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t


ctypedef uint32_t info_t
cdef object np_info = np.uint32
ctypedef np.uint32_t np_info_t


cdef class ParallelDelaunayD:

    cdef ParallelDelaunay_with_info_D[info_t] *T
    cdef object pts_total
    cdef int rank
    cdef int size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray[np.float64_t, ndim=1] le = None,
                  np.ndarray[np.float64_t, ndim=1] re = None,
                  object periodic=False, str unique_str="", int limit_mem=0):
        cdef np.uint32_t ndim = 0
        cdef cbool* per = NULL
        cdef double* ptr_le = NULL
        cdef double* ptr_re = NULL
        cdef object comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        cdef bytes py_bytes = unique_str.encode()
        cdef char* c_unique_str = py_bytes
        if self.rank == 0:
            ndim = le.size
            ptr_le = &le[0]
            ptr_re = &re[0]
            per = <cbool *>malloc(ndim*sizeof(cbool))
            if isinstance(periodic, pybool):
                for i in range(ndim):
                    per[i] = <cbool>periodic
            else:
                for i in range(ndim):
                    per[i] = <cbool>periodic[i]
        else:
            assert(le == None)
            assert(re == None)
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T = new ParallelDelaunay_with_info_D[info_t](
                ndim, ptr_le, ptr_re, per, limit_mem, c_unique_str)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def insert(self, np.ndarray[np.float64_t, ndim=2] pts = None):
        cdef np.uint32_t ndim = 0
        cdef np.uint64_t npts = 0
        cdef double *ptr_pts = NULL
        if self.rank == 0:
            ndim = pts.shape[1]
            assert(ndim == self.T.ndim)
            npts = pts.shape[0]
            ptr_pts = &pts[0,0]
        else:
            assert(pts == None)
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.insert(npts, ptr_pts)
        self.pts_total = pts

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def consolidate_vols(self):
        cdef np.ndarray[np.float64_t, ndim=1] vols
        vols = np.empty(self.T.npts_total, 'float64')
        with nogil, cython.boundscheck(False), cython.wraparound(False):
            self.T.consolidate_vols(&vols[0])
        return vols

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
