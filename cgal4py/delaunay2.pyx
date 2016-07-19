# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
"""
delaunay2.pyx

Wrapper for CGAL 2D Delaunay Triangulation
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
from cython.operator cimport preincrement, predecrement
from libc.stdint cimport uint32_t, uint64_t

cdef class Delaunay2_vertex:
    r"""Wrapper class for a triangulation vertex.

    Args:
        T (Delaunay2): Triangulation that this vertex belongs to.

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this vertex belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].All_verts_iter`): C++ vertex 
            object. Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].All_verts_iter x
    
    def __cinit__(self, Delaunay2 T):
        self.T = T.T

    def __richcmp__(Delaunay2_vertex self, Delaunay2_vertex solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the vertex is the infinite vertex.

        Returns:
            bool: True if the vertex is the infinite vertex, False otherwise.

        """
        return self.T.is_infinite(self.x)

    def increment(self):
        r"""Advance to the next vertex in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous vertex in the triangulation."""
        predecrement(self.x)

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
        the vertex."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
            self.x.point(&out[0])
            return out

    property index:
        r"""uint64: The index of the vertex point in the input array."""
        def __get__(self):
            cdef np.uint64_t out = self.x.info()
            return out

cdef class Delaunay2_vertex_range:
    r"""Wrapper class for iterating over a range of triangulation vertices
    
    Args:
        vstart (Delaunay2_vertex): The starting vertex.
        vstop (Delaunay2_vertex): Final vertex that will end the iteration.

    Attributes:
        x (Delaunay2_vertex): The current vertex.
        xstop (Delaunay2_vertex): Final vertex that will end the iteration.

    """
    cdef Delaunay2_vertex x
    cdef Delaunay2_vertex xstop
    def __cinit__(self, Delaunay2_vertex xstart, Delaunay2_vertex xstop):
        self.x = xstart
        self.xstop = xstop
        self.x.decrement()

    def __iter__(self):
        return self

    def __next__(self):
        self.x.increment()
        cdef Delaunay2_vertex out = self.x
        if self.x != self.xstop:
            return out
        else:
            raise StopIteration()


cdef class Delaunay2_cell:
    r"""Wrapper class for a triangulation cell.

    Args:
        T (Delaunay2): Triangulation that this cell belongs to.

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this cell belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].All_cells_iter`): C++ cell 
            object. Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].All_cells_iter x
    
    def __cinit__(self, Delaunay2 T):
        self.T = T.T

    def __richcmp__(Delaunay2_cell self, Delaunay2_cell solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the cell is incident to the infinite vertex.

        Returns:
            bool: True if the cell is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def increment(self):
        r"""Advance to the next cell in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous cell in the triangulation."""
        predecrement(self.x)


cdef class Delaunay2_cell_range:
    r"""Wrapper class for iterating over a range of triangulation cells.
    
    Args:
        xstart (Delaunay2_cell): The starting cell.
        xstop (Delaunay2_cell): Final cell that will end the iteration.

    Attributes:
        x (Delaunay2_cell): The current cell.
        xstop (Delaunay2_cell): Final cell that will end the iteration.

    """
    cdef Delaunay2_cell x
    cdef Delaunay2_cell xstop
    def __cinit__(self, Delaunay2_cell xstart, Delaunay2_cell xstop):
        self.x = xstart
        self.xstop = xstop
        self.x.decrement()

    def __iter__(self):
        return self

    def __next__(self):
        self.x.increment()
        cdef Delaunay2_cell out = self.x
        if self.x != self.xstop:
            return out
        else:
            raise StopIteration()


cdef class Delaunay2:
    r"""Wrapper class for a 2D Delaunay triangulation.

    Attributes:
        n (int): The number of points inserted into the triangulation.
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended.

    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.n = 0
        self.T = new Delaunay_with_info_2[uint32_t]()

    def write_to_file(self, fname):
        r"""Write the serialized tessellation information to a file.

        Args:
            fname (str): The full path to the file that the tessellation should 
                be written to.

        """
        cdef char* cfname = fname
        self.T.write_to_file(cfname)

    def read_from_file(self, fname):
        r"""Read serialized tessellation information from a file.

        Args:
            fname (str): The full path to the file that the tessellation should 
                be read from.

        """
        cdef char* cfname = fname
        self.T.read_from_file(cfname)
        self.n = self.num_finite_verts

    @property
    def num_finite_verts(self): 
        r"""int: The number of finite vertices in the triangulation."""
        return self.T.num_finite_verts()
    @property
    def num_finite_cells(self): 
        r"""int: The number of finite cells in the triangulation."""
        return self.T.num_finite_cells()
    @property
    def num_infinite_verts(self):
        r"""int: The number of infinite vertices in the triangulation."""
        return self.T.num_infinite_verts()
    @property
    def num_infinite_cells(self): 
        r"""int: The number of infinite cells in the triangulation."""
        return self.T.num_infinite_cells()
    @property
    def num_verts(self): 
        r"""int: The total number of vertices (Finite + infinite) in the 
        triangulation."""
        return self.T.num_verts()
    @property
    def num_cells(self): 
        r"""int: The total number of cells (finite + infinite) in the 
        triangulation."""
        return self.T.num_cells()

    def insert(self, np.ndarray[double, ndim=2, mode="c"] pts not None):
        r"""Insert points into the triangulation.

        Args:
            pts (:obj:`ndarray` of :obj:`float64`): Array of 2D cartesian 
                points to insert into the triangulation.

        """
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
        if self.n != self.num_finite_verts:
            print("There were {} duplicates".format(self.n-self.num_finite_verts))
        # assert(self.n == self.num_finite_verts)

    @property
    def all_verts_begin(self):
        r"""Delaunay2_vertex: Starting vertex for the triangulation."""
        cdef Delaunay_with_info_2[uint32_t].All_verts_iter it_begin
        it_begin = self.T.all_verts_begin()
        cdef Delaunay2_vertex v_begin = Delaunay2_vertex(self)
        v_begin.x = it_begin
        return v_begin
    @property
    def all_verts_end(self):
        r"""Delaunay2_vertex: Final vertex in the triangulation."""
        cdef Delaunay_with_info_2[uint32_t].All_verts_iter it_end
        it_end = self.T.all_verts_end()
        cdef Delaunay2_vertex v_end = Delaunay2_vertex(self)
        v_end.x = it_end
        return v_end
    @property
    def all_verts(self):
        r"""Delaunay2_vertex_range: Iterable for all vertices in the 
        triangulation."""
        return Delaunay2_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end)

    @property
    def all_cells_begin(self):
        r"""Delaunay2_cell: Starting cell for the triangulation."""
        cdef Delaunay_with_info_2[uint32_t].All_cells_iter it_begin
        it_begin = self.T.all_cells_begin()
        cdef Delaunay2_cell c_begin = Delaunay2_cell(self)
        c_begin.x = it_begin
        return c_begin
    @property
    def all_cells_end(self):
        r"""Delaunay2_cell: Finall cell in the triangulation."""
        cdef Delaunay_with_info_2[uint32_t].All_cells_iter it_end
        it_end = self.T.all_cells_end()
        cdef Delaunay2_cell c_end = Delaunay2_cell(self)
        c_end.x = it_end
        return c_end
    @property
    def all_cells(self):
        r"""Delaunay2_cell_range: Iterable for all cells in the
        triangulation."""
        return Delaunay2_cell_range(self.all_cells_begin,
                                    self.all_cells_end)

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

