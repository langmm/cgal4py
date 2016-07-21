# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
"""
delaunay3.pyx

Wrapper for CGAL 3D Delaunay Triangulation
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

cdef class Delaunay3_vertex:
    r"""Wrapper class for a triangulation vertex.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this vertex belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].Vertex`): C++ vertex object 
            Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].Vertex x

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     Delaunay_with_info_3[uint32_t].Vertex x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
                that this vertex belongs to. 
            x (:obj:`Delaunay_with_info_3[uint32_t].Vertex`): C++ vertex object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    def __richcmp__(Delaunay3_vertex self, Delaunay3_vertex solf, int op):
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

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
        of the vertex."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
            self.x.point(&out[0])
            return out

    property index:
        r"""uint64: The index of the vertex point in the input array."""
        def __get__(self):
            cdef np.uint64_t out = self.x.info()
            return out

    property volume:
        r"""float64: The volume of the dual Voronoi cell. If the volume is 
        infinite, -1.0 is returned."""
        def __get__(self):
            cdef np.float64_t out = self.T.dual_volume(self.x)
            return out

    def incident_cells(self):
        r"""Find cells that are incident to this vertex.

        Returns:
            Delaunay3_cell_vector: Iterator over cells incident to this vertex.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef Delaunay3_cell_vector out = Delaunay3_cell_vector()
        out.assign(self.T, it)
        return out

    def incident_vertices(self):
        r"""Find vertices that are adjacent to this vertex.

        Returns:
            Delaunay3_vertex_vector: Iterator over cells incident to this vertex.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef Delaunay3_vertex_vector out = Delaunay3_vertex_vector()
        out.assign(self.T, it)
        return out


cdef class Delaunay3_vertex_iter:
    r"""Wrapper class for a triangulation vertex iterator.

    Args:
        T (Delaunay3): Triangulation that this vertex belongs to.
        vert (:obj:`str`, optional): String specifying the vertex that 
            should be referenced. Valid options include: 
                'all_begin': The first vertex in an iteration over all vertices.  
                'all_end': The last vertex in an iteration over all vertices. 
 
    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this vertex belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].All_verts_iter`): C++ vertex 
            object. Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].All_verts_iter x

    def __cinit__(self, Delaunay3 T, str vert = None):
        self.T = T.T
        if vert == 'all_begin':
            self.x = self.T.all_verts_begin()
        elif vert == 'all_end':
            self.x = self.T.all_verts_end()

    def __richcmp__(Delaunay3_vertex_iter self, Delaunay3_vertex_iter solf, 
                    int op):
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

    property vertex:
        r"""Delaunay3_vertex: Corresponding vertex object."""
        def __get__(self):
            cdef Delaunay3_vertex out = Delaunay3_vertex()
            out.assign(self.T, Delaunay_with_info_3[uint32_t].Vertex(self.x))
            return out


cdef class Delaunay3_vertex_range:
    r"""Wrapper class for iterating over a range of triangulation vertices

    Args:
        xstart (Delaunay3_vertex_iter): The starting vertex.  
        xstop (Delaunay3_vertex_iter): Final vertex that will end the iteration. 
        finite (:obj:`bool`, optional): If True, only finite verts are 
            iterated over. Otherwise, all verts are iterated over. Defaults 
            False.

    Attributes:
        x (Delaunay3_vertex_iter): The current vertex. 
        xstop (Delaunay3_vertex_iter): Final vertex that will end the iteration. 
        finite (bool): If True, only finite verts are iterater over. Otherwise
            all verts are iterated over. 

    """
    cdef Delaunay3_vertex_iter x
    cdef Delaunay3_vertex_iter xstop
    cdef pybool finite
    def __cinit__(self, Delaunay3_vertex_iter xstart, 
                  Delaunay3_vertex_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite
        self.x.decrement()

    def __iter__(self):
        return self

    def __next__(self):
        self.x.increment()
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef Delaunay3_vertex out = self.x.vertex
        if self.x != self.xstop:
            return out
        else:
            raise StopIteration()

cdef class Delaunay3_vertex_vector:
    r"""Wrapper class for a vector of vertices.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_3[uint32_t].Vertex]`): Vector of C++ 
            vertices.
        n (int): The number of vertices in the vector.
        i (int): The index of the currect vertex.

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef vector[Delaunay_with_info_3[uint32_t].Vertex] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     vector[Delaunay_with_info_3[uint32_t].Vertex] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object.
                Direct interaction with this object is not recommended.
            v (:obj:`vector[Delaunay_with_info_3[uint32_t].Vertex]`): Vector of 
                C++ vertices.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef Delaunay3_vertex out
        if self.i < self.n:
            out = Delaunay3_vertex()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()


cdef class Delaunay3_cell:
    r"""Wrapper class for a triangulation cell.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this cell belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].Cell`): C++ cell object. 
            Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].Cell x

    def __richcmp__(Delaunay3_cell self, Delaunay3_cell solf, int op):
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

    def circumcenter(self):
        r"""Determines the circumcenter of the cell.

        Returns:
            :obj:`ndarray` of float64: x,y cartesian coordinates of the cell's
                circumcenter.

        """
        cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
        self.T.circumcenter(self.x, &out[0])
        return out


cdef class Delaunay3_cell_iter:
    r"""Wrapper class for a triangulation cell iteration.

    Args:
        T (Delaunay3): Triangulation that this cell belongs to. 
        cell (:obj:`str`, optional): String specifying the cell that
            should be referenced. Valid options include: 
                'all_begin': The first cell in an iteration over all cells. 
                'all_end': The last cell in an iteration over all cells.
    
    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this cell belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].All_cells_iter`): C++ cell
            object. Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].All_cells_iter x

    def __cinit__(self, Delaunay3 T, str cell = None):
        self.T = T.T
        if cell == 'all_begin':
            self.x = self.T.all_cells_begin()
        elif cell == 'all_end':
            self.x = self.T.all_cells_end()

    def __richcmp__(Delaunay3_cell_iter self, Delaunay3_cell_iter solf, int op):
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

    property cell:
        r"""Delaunay3_cell: Corresponding cell object."""
        def __get__(self):
            cdef Delaunay3_cell out = Delaunay3_cell()
            out.T = self.T
            out.x = Delaunay_with_info_3[uint32_t].Cell(self.x)
            return out


cdef class Delaunay3_cell_range:
    r"""Wrapper class for iterating over a range of triangulation cells.

    Args:
        xstart (Delaunay3_cell_iter): The starting cell. 
        xstop (Delaunay3_cell_iter): Final cell that will end the iteration. 
        finite (:obj:`bool`, optional): If True, only finite cells are 
            iterated over. Otherwise, all cells are iterated over. Defaults
            to False.  

    Attributes:
        x (Delaunay3_cell_iter): The current cell. 
        xstop (Delaunay3_cell_iter): Final cell that will end the iteration. 
        finite (bool): If True, only finite cells are iterated over. Otherwise, 
            all cells are iterated over.   

    """
    cdef Delaunay3_cell_iter x
    cdef Delaunay3_cell_iter xstop
    cdef pybool finite
    def __cinit__(self, Delaunay3_cell_iter xstart, Delaunay3_cell_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite
        self.x.decrement()

    def __iter__(self):
        return self

    def __next__(self):
        self.x.increment()
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef Delaunay3_cell out = self.x.cell
        if self.x != self.xstop:
            return out
        else:
            raise StopIteration()

cdef class Delaunay3_cell_vector:
    r"""Wrapper class for a vector of cells.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_3[uint32_t].Cell]`): Vector of C++ 
            cells.
        n (int): The number of cells in the vector.
        i (int): The index of the currect cell.

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef vector[Delaunay_with_info_3[uint32_t].Cell] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     vector[Delaunay_with_info_3[uint32_t].Cell] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object.
                Direct interaction with this object is not recommended.
            v (:obj:`vector[Delaunay_with_info_3[uint32_t].Cell]`): Vector of 
                C++ cells.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef Delaunay3_cell out
        if self.i < self.n:
            out = Delaunay3_cell()
            out.T = self.T
            out.x = self.v[self.i]
            self.i += 1
            return out
        else:
            raise StopIteration()


cdef class Delaunay3:
    r"""Wrapper class for a 3D Delaunay triangulation.

    Attributes:
        n (int): The number of points inserted into the triangulation.
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 

    """
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.n = 0
        self.T = new Delaunay_with_info_3[uint32_t]()

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
    def num_finite_edges(self):
        r"""int: The number of finite edges in the triangulation."""
        return self.T.num_finite_edges()
    @property
    def num_finite_cells(self):
        r"""int: The number of finite cells in the triangulation."""
        return self.T.num_finite_cells()
    @property
    def num_infinite_verts(self):
        r"""int: The number of infinite vertices in the triangulation."""
        return self.T.num_infinite_verts()
    @property
    def num_infinite_edges(self):
        r"""int: The number of infinite edges in the triangulation."""
        return self.T.num_infinite_edges()
    @property
    def num_infinite_cells(self):
        r"""int: The number of infinite cells in the triangulation."""
        return self.T.num_infinite_cells()
    @property
    def num_verts(self): 
        r"""int: The total number of vertices (finite + infinite) in the 
        triangulation."""
        return self.T.num_verts()
    @property
    def num_edges(self):
        r"""int: The total number of edges (finite + infinite) in the 
        triangulation."""
        return self.T.num_edges()
    @property
    def num_cells(self):
        r"""int: The total number of cells (finite + infinite) in the 
        triangulation."""
        return self.T.num_cells()

    def insert(self, np.ndarray[double, ndim=2, mode="c"] pts not None):
        r"""Insert points into the triangulation.

        Args:
            pts (:obj:`ndarray` of :obj:`float64`): Array of 3D cartesian 
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
        assert(m == 3)
        self.T.insert(&pts[0,0], &idx[0], <uint32_t>Nnew)
        self.n += Nnew
        if self.n != self.num_finite_verts:
            print "There were {} duplicates".format(self.n-self.num_finite_verts)
        # assert(self.n == self.num_finite_verts)

    @property
    def all_verts_begin(self):
        r"""Delaunay3_vertex_iter: Starting vertex for all vertices in the 
        triangulation."""
        return Delaunay3_vertex_iter(self, 'all_begin')
    @property
    def all_verts_end(self):
        r"""Delaunay3_vertex_iter: Final vertex for all vertices in the 
        triangulation."""
        return Delaunay3_vertex_iter(self, 'all_end')
    @property
    def all_verts(self):
        r"""Delaunay3_vertex_range: Iterable for all vertices in the 
        triangulation."""
        return Delaunay3_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end)
    @property
    def finite_verts(self):
        r"""Delaunay3_vertex_range: Iterable for finite vertices in the 
        triangulation."""
        return Delaunay3_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end, finite = True)

    @property
    def all_cells_begin(self):
        r"""Delaunay3_cell_iter: Starting cell for all cells in the triangulation."""
        return Delaunay3_cell_iter(self, 'all_begin')
    @property
    def all_cells_end(self):
        r"""Delaunay3_cell_iter: Finall cell for all cells in the triangulation."""
        return Delaunay3_cell_iter(self, 'all_end')
    @property
    def all_cells(self):
        r"""Delaunay3_cell_range: Iterable for all cells in the
        triangulation."""
        return Delaunay3_cell_range(self.all_cells_begin,
                                    self.all_cells_end)
    @property
    def finite_cells(self):
        r"""Delaunay3_cell_range: Iterable for finite cells in the
        triangulation."""
        return Delaunay3_cell_range(self.all_cells_begin,
                                    self.all_cells_end, finite = True)

    def nearest_vertex(self, np.ndarray[np.float64_t, ndim=1] x):
        r"""Determine which vertex is closes to a given set of x,y coordinates.

        Args:
            x (:obj:`ndarray` of float64): x,y coordinates. 

        Returns: 
            Delaunay3_vertex: Vertex closest to x. 

        """
        cdef Delaunay_with_info_3[uint32_t].Vertex vc
        vc = self.T.nearest_vertex(&x[0])
        cdef Delaunay3_vertex v = Delaunay3_vertex()
        v.assign(self.T, vc)
        return v

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

