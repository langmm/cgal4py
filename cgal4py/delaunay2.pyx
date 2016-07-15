"""
delaunay.pyx

Wrapper for CGAL Triangulation
"""

import cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from cython.operator cimport dereference
from cython.operator cimport preincrement
from libc.stdint cimport uint32_t, uint64_t

cdef class Delaunay2:
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.n = 0
        self.T = new Delaunay_with_info_2[uint32_t]()

    def write_to_file(self, fname):
        cdef char* cfname = fname
        self.T.write_to_file(cfname)

    def read_from_file(self, fname):
        cdef char* cfname = fname
        self.T.read_from_file(cfname)
        self.n = self.num_verts()

    def num_verts(self): return self.T.num_verts()
    # def num_edges(self): return self.T.num_edges()
    def num_cells(self): return self.T.num_cells()

    def insert(self, np.ndarray[double, ndim=2, mode="c"] pts not None):
        if pts.shape[0] != 0:
            self._insert(pts)
    cdef void _insert(self, np.ndarray[double, ndim=2, mode="c"] pts):
        cdef int Nold, Nnew, m
        Nold = self.n
        Nnew, m = pts.shape[0], pts.shape[1]
        cdef np.ndarray[uint32_t, ndim=1] idx
        idx = np.arange(Nold, Nold+Nnew).astype('uint32')
        assert(m == 2)
        self.T.insert(&pts[0,0], &idx[0], <uint32_t>Nnew)
        self.n += Nnew
        if self.n != self.num_verts():
            print "There were {} duplicates".format(self.n-self.num_verts())
        # assert(self.n == self.num_verts())

    def edge_info(self, max_incl, idx):
        return self._edge_info(max_incl, idx)
    cdef object _edge_info(self, int max_incl, np.uint64_t[:] idx):
        cdef object edge_list = []
        cdef vector[pair[uint32_t,uint32_t]] edge_vect
        cdef int j
        cdef uint32_t i1, i2
        self.T.edge_info(edge_vect)
        for j in xrange(<int>edge_vect.size()):
            i1 = edge_vect[j].first
            i2 = edge_vect[j].second
            if idx[i2] < idx[i1]:
                i1, i2 = i2, i1
            if i1 < (<uint32_t>max_incl):
                edge_list.append(np.array([i1,i2],'int64'))
        return np.array(edge_list)

    def outgoing_points(self, 
                        np.ndarray[double, ndim=1] left_edge,
                        np.ndarray[double, ndim=1] right_edge,
                        cbool periodic, object neighbors, int num_leaves):
        return self._outgoing_points(left_edge, right_edge, periodic,
                                     neighbors, num_leaves)
    cdef object _outgoing_points(self, 
                                 np.ndarray[double, ndim=1] left_edge,
                                 np.ndarray[double, ndim=1] right_edge,
                                 cbool periodic, object neighbors, int num_leaves):
        cdef int i, j, k
        cdef vector[uint32_t] lr, lx, ly, lz, rx, ry, rz, alln
        self.T.outgoing_points(&left_edge[0], &right_edge[0], periodic,
                               lx, ly, rx, ry, alln)
        # Get counts to preallocate arrays
        cdef object hvall
        cdef np.ndarray[np.uint32_t] Nind = np.zeros(num_leaves, 'uint32')
        cdef np.uint32_t iN
        Nind += <np.uint32_t>alln.size()
        for i, lr in enumerate([lx, ly]):
            iN = <np.uint32_t>lr.size()
            for k in neighbors[i]['left']+neighbors[i]['left_periodic']:
                Nind[k] += iN
        for i, lr in enumerate([rx, ry]):
            iN = <np.uint32_t>lr.size()
            for k in neighbors[i]['right']+neighbors[i]['right_periodic']:        
                Nind[k] += iN
        hvall = [np.zeros(Nind[j], 'uint32') for j in xrange(num_leaves)]

        # Transfer values
        cdef np.ndarray[np.uint32_t] Cind = np.zeros(num_leaves, 'uint32')
        cdef np.ndarray[np.uint32_t] lr_arr
        iN = alln.size()
        lr_arr = np.array([alln[j] for j in xrange(<int>iN)], 'uint32')
        for k in xrange(num_leaves):
            hvall[k][Cind[k]:(Cind[k]+iN)] = lr_arr
            Cind[k] += iN
        for i, lr in enumerate([lx, ly]):
            iN = <np.uint32_t>lr.size()
            lr_arr = np.array([lr[j] for j in xrange(<int>iN)], 'uint32')
            for k in neighbors[i]['left']+neighbors[i]['left_periodic']:
                hvall[k][Cind[k]:(Cind[k]+iN)] = lr_arr
                Cind[k] += iN
        for i, lr in enumerate([rx, ry]):
            iN = <np.uint32_t>lr.size()
            lr_arr = np.array([lr[j] for j in xrange(<int>iN)], 'uint32')
            for k in neighbors[i]['right']+neighbors[i]['right_periodic']:
                hvall[k][Cind[k]:(Cind[k]+iN)] = lr_arr
                Cind[k] += iN

        # Find unique indices
        for k in xrange(num_leaves):
            hvall[k] = np.unique(hvall[k])

        return hvall

    def voronoi_volumes(self, max_idx):
        cdef np.ndarray[np.float64_t, ndim=1] vol
        vol = -999*np.ones(max_idx, 'float64')
        self._voronoi_volumes(max_idx, vol)
        return vol
    cdef void _voronoi_volumes(self, int max_idx, np.float64_t[:] vol):
        cdef vector[pair[uint32_t,double]] out
        voronoi_areas(self.T, out)
        cdef int i
        cdef int count = 0
        for i in xrange(<int>out.size()):
            if <int>out[i].first < max_idx:
                vol[out[i].first] = out[i].second
                count += 1

