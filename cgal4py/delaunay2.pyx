"""
delaunay2.pyx

Wrapper for CGAL 2D Delaunay Triangulation
"""

import cython

import numpy as np
cimport numpy as np

from . import plot

from functools import wraps

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

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this vertex belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].Vertex`): C++ vertex 
            object. Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].Vertex x

    cdef void assign(self, Delaunay_with_info_2[uint32_t] *T,
                     Delaunay_with_info_2[uint32_t].Vertex x):
        r"""Assign C++ objects to attributes.

        Args:
            T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
                that this vertex belongs to.
            x (:obj:`Delaunay_with_info_2[uint32_t].Vertex`): C++ vertex 
                object. Direct interaction with this object is not recommended.

        """
        self.T = T
        self.x = x

    def __repr__(self):
        return "Delaunay2_vertex[{} at {:+7.2e},{:+7.2e}]".format(
            self.index, *list(self.point))

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

    def set_point(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Set this vertex's coordinates.

        Args:
            pos (:obj:`ndarray` of float64): new x,y coordinates for this vertex.

        """
        self.T.updated = <cbool>True
        assert(len(pos) == 2)
        self.x.set_point(&pos[0])

    def set_cell(self, Delaunay2_cell c):
        r"""Assign this vertex's designated cell.

        Args:
            c (Delaunay2_cell): Cell that will be assigned as designated cell.

        """
        self.T.updated = <cbool>True
        self.x.set_cell(c.x)

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y) coordinates of 
        the vertex."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
            if self.is_infinite():
                out[:] = np.float('inf')
            else:
                self.x.point(&out[0])
            return out

    property index:
        r"""uint32: The index of the vertex point in the input array."""
        def __get__(self):
            cdef np.uint32_t out
            if self.is_infinite():
                out = np.iinfo(np.uint32).max
            else:
                out = self.x.info()
            return out

    property dual_volume:
        r"""float64: The area of the dual Voronoi cell. If the area is 
        infinite, -1.0 is returned."""
        def __get__(self):
            cdef np.float64_t out = self.T.dual_area(self.x)
            return out

    property cell:
        r"""Delaunay2_cell: The cell assigned to this vertex."""
        def __get__(self):
            cdef Delaunay_with_info_2[uint32_t].Cell c
            c = self.x.cell()
            cdef Delaunay2_cell out = Delaunay2_cell()
            out.assign(self.T, c)
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this vertex.

        Returns:
            Delaunay2_vertex_vector: Iterator over vertices incident to this 
                vertex.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef Delaunay2_vertex_vector out = Delaunay2_vertex_vector()
        out.assign(self.T, it)
        return out
        
    def incident_edges(self):
        r"""Find edges that are incident to this vertex.

        Returns:
            Delaunay2_edge_vector: Iterator over edges incident to this vertex.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef Delaunay2_edge_vector out = Delaunay2_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this vertex.

        Returns:
            Delaunay2_cell_vector: Iterator over cells incident to this vertex.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef Delaunay2_cell_vector out = Delaunay2_cell_vector()
        out.assign(self.T, it)
        return out


cdef class Delaunay2_vertex_iter:
    r"""Wrapper class for a triangulation vertex iterator.

    Args:
        T (Delaunay2): Triangulation that this vertex belongs to.
        vert (:obj:`str`, optional): String specifying the vertex that 
            should be referenced. Valid options include:
                'all_begin': The first vertex in an iteration over all vertices.
                'all_end': The last vertex in an iteration over all vertices.

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this vertex belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].All_verts_iter`): C++ vertex 
            iteration object. Direct interaction with this object is not 
            recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].All_verts_iter x
    
    def __cinit__(self, Delaunay2 T, str vert = None):
        self.T = T.T
        if vert == 'all_begin':
            self.x = self.T.all_verts_begin()
        elif vert == 'all_end':
            self.x = self.T.all_verts_end()

    def __richcmp__(Delaunay2_vertex_iter self, Delaunay2_vertex_iter solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def increment(self):
        r"""Advance to the next vertex in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous vertex in the triangulation."""
        predecrement(self.x)

    property vertex:
        r"""Delaunay2_vertex: The corresponding vertex object."""
        def __get__(self):
            cdef Delaunay2_vertex out = Delaunay2_vertex()
            out.assign(self.T, Delaunay_with_info_2[uint32_t].Vertex(self.x))
            return out

cdef class Delaunay2_vertex_vector:
    r"""Wrapper class for a vector of vertices. 

    Attributes: 
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 
        v (:obj:`vector[Delaunay_with_info_2[uint32_t].Vertex]`): Vector of C++ 
            vertices. 
        n (int): The number of vertices in the vector. 
        i (int): The index of the currect vertex. 

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef vector[Delaunay_with_info_2[uint32_t].Vertex] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_2[uint32_t] *T,
                     vector[Delaunay_with_info_2[uint32_t].Vertex] v):
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
        cdef Delaunay2_vertex out
        if self.i < self.n:
            out = Delaunay2_vertex()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef Delaunay2_vertex out
        if isinstance(i, int):
            out = Delaunay2_vertex()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("Delaunay2_vertex_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class Delaunay2_vertex_range:
    r"""Wrapper class for iterating over a range of triangulation vertices
    
    Args:
        vstart (Delaunay2_vertex_iter): The starting vertex.
        vstop (Delaunay2_vertex_iter): Final vertex that will end the iteration.
        finite (:obj:`bool`, optional): If True, only finite verts are
            iterated over. Otherwise, all verts are iterated over. Defaults 
            to False.

    Attributes:
        x (Delaunay2_vertex_iter): The current vertex.
        xstop (Delaunay2_vertex_iter): Final vertex that will end the iteration.
        finite (bool): If True, only finite verts are iterater over. Otherwise 
            all verts are iterated over.

    """
    cdef Delaunay2_vertex_iter x
    cdef Delaunay2_vertex_iter xstop
    cdef pybool finite
    def __cinit__(self, Delaunay2_vertex_iter xstart, Delaunay2_vertex_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.T.is_infinite(self.x.x):
                self.x.increment()
        cdef Delaunay2_vertex out
        if self.x != self.xstop:
            out = self.x.vertex
            self.x.increment()
            return out
        else:
            raise StopIteration()


cdef class Delaunay2_edge:
    r"""Wrapper class for a triangulation edge.

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this edge belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].Edge`): C++ edge 
            object. Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].Edge x

    cdef void assign(self, Delaunay_with_info_2[uint32_t] *T,
                     Delaunay_with_info_2[uint32_t].Edge x):
        r"""Assign C++ objects to attributes.

        Args:
            T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
                that this edge belongs to.
            x (:obj:`Delaunay_with_info_2[uint32_t].Edge`): C++ edge 
                object. Direct interaction with this object is not recommended.

        """
        self.T = T
        self.x = x

    @staticmethod
    def from_cell(Delaunay2_cell c, int i):
        r"""Construct an edges from a cell and index of the vertex opposite the 
        edge.

        Args:
            c (Delaunay2_cell): Cell
            i (int): Index of vertex opposite the desired edge in c.

        Returns:
            Delaunay2_edge: Edge incident to c and opposite vertex i of c.

        """
        cdef Delaunay2_edge out = Delaunay2_edge()
        cdef Delaunay_with_info_2[uint32_t].Edge e
        e = Delaunay_with_info_2[uint32_t].Edge(c.x, i)
        out.assign(c.T, e)
        return out

    def __repr__(self):
        return "Delaunay2_edge[{},{}]".format(repr(self.vertex1), 
                                              repr(self.vertex2))

    def __richcmp__(Delaunay2_edge self, Delaunay2_edge solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the edge is incident to the infinite vertex.

        Returns:
            bool: True if the edge is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    property vertex1:
        r"""Delaunay2_vertex: The first vertex in the edge."""
        def __get__(self):
            cdef Delaunay_with_info_2[uint32_t].Vertex x = self.x.v1()
            cdef Delaunay2_vertex out = Delaunay2_vertex()
            out.assign(self.T, x)
            return out

    property vertex2:
        r"""Delaunay2_vertex: The second vertex in the edge."""
        def __get__(self):
            cdef Delaunay_with_info_2[uint32_t].Vertex x = self.x.v2()
            cdef Delaunay2_vertex out = Delaunay2_vertex()
            out.assign(self.T, x)
            return out

    property length:
        r"""float64: The length of the edge. If infinite, -1 is returned"""
        def __get__(self):
            cdef np.float64_t out = self.T.length(self.x)
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this edge.

        Returns:
            Delaunay2_vertex_vector: Iterator over vertices incident to this 
                edge.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        # it.push_back(self.x.v1())
        # it.push_back(self.x.v2())
        cdef Delaunay2_vertex_vector out = Delaunay2_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this edge.

        Returns:
            Delaunay2_edge_vector: Iterator over edges incident to this edge.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef Delaunay2_edge_vector out = Delaunay2_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this edge.

        Returns:
            Delaunay2_cell_vector: Iterator over cells incident to this edge.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef Delaunay2_cell_vector out = Delaunay2_cell_vector()
        out.assign(self.T, it)
        return out

    def flip(self):
        r"""Flip this edge to the other diagonal of the quadrilateral formed by 
        the two cells incident to this edge after first testing that the edge 
        can be flipped.

        Returns:
            bool: True if the edge could be flipped, False otherwise.

        """
        self.T.updated = <cbool>True
        return self.T.flip(self.x)

    def flip_flippable(self):
        r"""Flip this edge to the other diagonal of the quadrilateral formed by 
        the two cells incident to this edge. The edge is assumed flippable to 
        save time.
        """
        self.T.updated = <cbool>True
        self.T.flip_flippable(self.x)

cdef class Delaunay2_edge_iter:
    r"""Wrapper class for a triangulation edge iterator.

    Args:
        T (Delaunay2): Triangulation that this edge belongs to.
        edge (:obj:`str`, optional): String specifying the edge that 
            should be referenced. Valid options include:
                'all_begin': The first edge in an iteration over all edges.
                'all_end': The last edge in an iteration over all edges.

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this edge belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].All_edges_iter`): C++ edge 
            iteration object. Direct interaction with this object is not 
            recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].All_edges_iter x
    
    def __cinit__(self, Delaunay2 T, str edge = None):
        self.T = T.T
        if edge == 'all_begin':
            self.x = self.T.all_edges_begin()
        elif edge == 'all_end':
            self.x = self.T.all_edges_end()

    def __richcmp__(Delaunay2_edge_iter self, Delaunay2_edge_iter solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def increment(self):
        r"""Advance to the next edge in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous edge in the triangulation."""
        predecrement(self.x)

    property edge:
        r"""Delaunay2_edge: The corresponding edge object."""
        def __get__(self):
            cdef Delaunay2_edge out = Delaunay2_edge()
            out.assign(self.T, Delaunay_with_info_2[uint32_t].Edge(self.x))
            return out

cdef class Delaunay2_edge_vector:
    r"""Wrapper class for a vector of edges.

    Attributes: 
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 
        v (:obj:`vector[Delaunay_with_info_2[uint32_t].Edge]`): Vector of C++ 
            edges.
        n (int): The number of edges in the vector. 
        i (int): The index of the currect edge.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef vector[Delaunay_with_info_2[uint32_t].Edge] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_2[uint32_t] *T,
                     vector[Delaunay_with_info_2[uint32_t].Edge] v):
        r"""Assign C++ attributes. 

        Args: 
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object. 
                Direct interaction with this object is not recommended. 
            v (:obj:`vector[Delaunay_with_info_3[uint32_t].Edge]`): Vector of 
                C++ edges. 

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef Delaunay2_edge out
        if self.i < self.n:
            out = Delaunay2_edge()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef Delaunay2_edge out
        if isinstance(i, int):
            out = Delaunay2_edge()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("Delaunay2_edge_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class Delaunay2_edge_range:
    r"""Wrapper class for iterating over a range of triangulation edges.
    
    Args:
        vstart (Delaunay2_edge_iter): The starting edge.
        vstop (Delaunay2_edge_iter): Final edge that will end the iteration.
        finite (:obj:`bool`, optional): If True, only finite edges are
            iterated over. Otherwise, all edges are iterated over. Defaults 
            to False.

    Attributes:
        x (Delaunay2_edge_iter): The current edge.
        xstop (Delaunay2_edge_iter): Final edge that will end the iteration.
        finite (bool): If True, only finite edges are iterater over. Otherwise 
            all edges are iterated over.

    """
    cdef Delaunay2_edge_iter x
    cdef Delaunay2_edge_iter xstop
    cdef pybool finite
    def __cinit__(self, Delaunay2_edge_iter xstart, Delaunay2_edge_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.T.is_infinite(self.x.x):
                self.x.increment()
        cdef Delaunay2_edge out
        if self.x != self.xstop:
            out = self.x.edge
            self.x.increment()
            return out
        else:
            raise StopIteration()


cdef class Delaunay2_cell:
    r"""Wrapper class for a triangulation cell.

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this cell belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].Cell`): C++ cell object.
            Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].Cell x
    
    cdef void assign(self, Delaunay_with_info_2[uint32_t] *T,
                     Delaunay_with_info_2[uint32_t].Cell x):
        r"""Assign C++ objects to attributes.

        Args:
            T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
                that this edge belongs to.
            x (:obj:`Delaunay_with_info_2[uint32_t].Cell`): C++ cell 
                object. Direct interaction with this object is not recommended.

        """
        self.T = T
        self.x = x

    def __richcmp__(Delaunay2_cell self, Delaunay2_cell solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def __repr__(self):
        return "Delaunay2_cell[{},{},{}]".format(repr(self.vertex(0)),
                                                 repr(self.vertex(1)),
                                                 repr(self.vertex(2)))

    def is_infinite(self):
        r"""Determine if the cell is incident to the infinite vertex.

        Returns:
            bool: True if the cell is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def vertex(self, int i):
        r"""Find the ith vertex that is incident to this cell.

        Args:
            i (int): The index of the vertex that should be returned.

        Returns:
            Delaunay2_vertex: The ith vertex incident to this cell.

        """
        cdef Delaunay_with_info_2[uint32_t].Vertex v
        v = self.x.vertex(i)
        cdef Delaunay2_vertex out = Delaunay2_vertex()
        out.assign(self.T, v)
        return out

    def has_vertex(self, Delaunay2_vertex v, pybool return_index = False):
        r"""Determine if a vertex belongs to this cell.

        Args:
            v (Delaunay2_vertex): Vertex to test ownership for.
            return_index (:obj:`bool`, optional): If True, the index of the 
                vertex within the cell is returned in the event that it is a 
                vertex of the cell. Otherwise, the index is not returned.

        Returns:
            bool: True if the vertex is part of the cell, False otherwise. In 
                the event that `return_index = True` and the vertex is a part of 
                the cell, an integer specifying the index of the vertex within 
                the cell will be returned instead.

        """
        cdef int i = -1
        cdef cbool out
        if return_index:
            out = self.x.has_vertex(v.x, &i)
            return i
        else:
            out = self.x.has_vertex(v.x)
            return <pybool>out

    def ind_vertex(self, Delaunay2_vertex v):
        r"""Determine the index of a vertex within a cell.

        Args:
            v (Delaunay2_vertex): Vertex to find index for.
        
        Returns:
            int: Index of vertex within the cell.

        """
        return self.x.ind(v.x)

    def neighbor(self, int i):
        r"""Find the neighboring cell opposite the ith vertex of this cell. 

        Args:
            i (int): The index of the neighboring cell that should be returned.

        Returns:
            Delaunay2_cell: The neighboring cell opposite the ith vertex.

        """
        cdef Delaunay_with_info_2[uint32_t].Cell v
        v = self.x.neighbor(i)
        cdef Delaunay2_cell out = Delaunay2_cell()
        out.assign(self.T, v)
        return out

    def has_neighbor(self, Delaunay2_cell v, pybool return_index = False):
        r"""Determine if a cell is a neighbor to this cell.

        Args:
            v (Delaunay2_cell): Cell to test as a neighbor.
            return_index (:obj:`bool`, optional): If True, the index of the 
                neighbor within the cell is returned in the event that it is a 
                neighbor of the cell. Otherwise, the index is not returned.

        Returns:
            bool: True if the given cell is a neighbor, False otherwise. In 
                the event that `return_index = True` and the v is a neighbor of 
                this cell, an integer specifying the index of the neighbor to 
                will be returned instead.

        """
        cdef int i = -1
        cdef cbool out = self.x.has_neighbor(v.x, &i)
        if out and return_index:
            return i
        else:
            return <pybool>out

    def ind_neighbor(self, Delaunay2_cell v):
        r"""Determine the index of a neighboring cell.

        Args:
            v (Delaunay2_cell): Neighboring cell to find index for.
        
        Returns:
            int: Index of vertex opposite to neighboring cell.

        """
        return self.x.ind(v.x)

    def set_vertex(self, int i, Delaunay2_vertex v):
        r"""Set the ith vertex of this cell.

        Args:
            i (int): Index of this cell's vertex that should be set.
            v (Delauany2_vertex): Vertex to set ith vertex of this cell to.

        """
        self.T.updated = <cbool>True
        self.x.set_vertex(i, v.x)

    def set_vertices(self, Delaunay2_vertex v1, Delaunay2_vertex v2, 
                     Delaunay2_vertex v3):
        r"""Set this cell's vertices.

        Args:
            v1 (Delaunay2_vertex): 1st vertex of cell.
            v2 (Delaunay2_vertex): 2nd vertex of cell.
            v3 (Delaunay2_vertex): 3rd vertex of cell.

        """
        self.T.updated = <cbool>True
        self.x.set_vertices(v1.x, v2.x, v3.x)

    def reset_vertices(self):
        r"""Reset all of this cell's vertices."""
        self.T.updated = <cbool>True
        self.x.set_vertices()

    def set_neighbor(self, int i, Delaunay2_cell n):
        r"""Set the ith neighboring cell of this cell.

        Args:
            i (int): Index of this cell's neighbor that should be set.
            n (Delaunay2_cell): Cell to set ith neighbor of this cell to.

        """
        self.T.updated = <cbool>True
        self.x.set_neighbor(i, n.x)

    def set_neighbors(self, Delaunay2_cell c1, Delaunay2_cell c2, 
                      Delaunay2_cell c3):
        r"""Set this cell's neighboring cells.

        Args:
            c1 (Delaunay2_cell): 1st neighboring cell.
            c2 (Delaunay2_cell): 2nd neighboring cell.
            c3 (Delaunay2_cell): 3rd neighboring cell.

        """
        self.T.updated = <cbool>True
        self.x.set_neighbors(c1.x, c2.x, c3.x)

    def reset_neighbors(self):
        r"""Reset all of this cell's neighboring cells."""
        self.T.updated = <cbool>True
        self.x.set_neighbors()

    def reorient(self):
        r"""Change the vertex order so that ccw and cw are switched."""
        self.T.updated = <cbool>True
        self.x.reorient()

    def ccw_permute(self):
        r"""Bring the last vertex to the front of the vertex order."""
        self.T.updated = <cbool>True
        self.x.ccw_permute()
        
    def cw_permute(self):
        r"""Put the 1st vertex at the end of the vertex order."""
        self.T.updated = <cbool>True
        self.x.cw_permute()

    property circumcenter:
        r""":obj:`ndarray` of float64: x,y coordinates of cell circumcenter."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(2, 'float64')
            self.T.circumcenter(self.x, &out[0])
            return out

    property dimension:
        r"""int: The number of dimensions that this cell occupies."""
        def __get__(self):
            return self.x.dimension()

    def incident_vertices(self):
        r"""Find vertices that are incident to this cell.

        Returns:
            Delaunay2_vertex_vector: Iterator over vertices incident to this 
                cell.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef Delaunay2_vertex_vector out = Delaunay2_vertex_vector()
        out.assign(self.T, it)
        return out
        
    def incident_edges(self):
        r"""Find edges that are incident to this cell.

        Returns:
            Delaunay2_edge_vector: Iterator over edges incident to this cell.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef Delaunay2_edge_vector out = Delaunay2_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this cell.

        Returns:
            Delaunay2_cell_vector: Iterator over cells incident to thiscell.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef Delaunay2_cell_vector out = Delaunay2_cell_vector()
        out.assign(self.T, it)
        return out

    def side_of_circle(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Determine where a point is with repect to this cell's 
            circumcircle.

        Args:
            pos (:obj:`ndarray` of np.float64): x,y coordinates.

        Returns:
            int: -1, 0, or 1 if `pos` is within, on, or inside this cell's 
                circumcircle respectively.

        """
        return self.T.side_of_oriented_circle(self.x, &pos[0])


cdef class Delaunay2_cell_iter:
    r"""Wrapper class for a triangulation cell.

    Args:
        T (Delaunay2): Triangulation that this cell belongs to.
        cell (:obj:`str`, optional): String specifying the cell that 
            should be referenced. Valid options include:
                'all_begin': The first cell in an iteration over all cells.
                'all_end': The last cell in an iteration over all cells.

    Attributes:
        T (:obj:`Delaunay_with_info_2[uint32_t]`): C++ Triangulation object 
            that this cell belongs to.
        x (:obj:`Delaunay_with_info_2[uint32_t].All_cells_iter`): C++ cell 
            object. Direct interaction with this object is not recommended.

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef Delaunay_with_info_2[uint32_t].All_cells_iter x
    
    def __cinit__(self, Delaunay2 T, str cell = None):
        self.T = T.T
        if cell == 'all_begin':
            self.x = self.T.all_cells_begin()
        elif cell == 'all_end':
            self.x = self.T.all_cells_end()

    def __richcmp__(Delaunay2_cell_iter self, Delaunay2_cell_iter solf, int op):
        if (op == 2): 
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def increment(self):
        r"""Advance to the next cell in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous cell in the triangulation."""
        predecrement(self.x)

    property cell:
        r"""Delaunay2_cell: Corresponding cell object."""
        def __get__(self):
            cdef Delaunay2_cell out = Delaunay2_cell()
            out.T = self.T
            out.x = Delaunay_with_info_2[uint32_t].Cell(self.x)
            return out


cdef class Delaunay2_cell_vector:
    r"""Wrapper class for a vector of cells. 

    Attributes: 
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 
        v (:obj:`vector[Delaunay_with_info_3[uint32_t].Cell]`): Vector of C++ 
            cells. 
        n (int): The number of cells in the vector. 
        i (int): The index of the currect cell. 

    """
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef vector[Delaunay_with_info_2[uint32_t].Cell] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_2[uint32_t] *T,
                     vector[Delaunay_with_info_2[uint32_t].Cell] v):
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
        cdef Delaunay2_cell out
        if self.i < self.n:
            out = Delaunay2_cell()
            out.T = self.T
            out.x = self.v[self.i]
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef Delaunay2_cell out
        if isinstance(i, int):
            out = Delaunay2_cell()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("Delaunay2_cell_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class Delaunay2_cell_range:
    r"""Wrapper class for iterating over a range of triangulation cells.
    
    Args:
        xstart (Delaunay2_cell_iter): The starting cell.
        xstop (Delaunay2_cell_iter): Final cell that will end the iteration.
        finite (:obj:`bool`, optional): If True, only finite cells are
            iterated over. Otherwise, all cells are iterated over. Defaults 
            to False.

    Attributes:
        x (Delaunay2_cell_iter): The current cell.
        xstop (Delaunay2_cell_iter): Final cell that will end the iteration.
        finite (bool): If True, only finite cells are iterated over. Otherwise, 
            all cells are iterated over.

    """
    cdef Delaunay2_cell_iter x
    cdef Delaunay2_cell_iter xstop
    cdef pybool finite
    def __cinit__(self, Delaunay2_cell_iter xstart, Delaunay2_cell_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.T.is_infinite(self.x.x):
                self.x.increment()
        cdef Delaunay2_cell out
        if self.x != self.xstop:
            out = self.x.cell
            self.x.increment()
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

    _cache_to_clear_on_update = {}

    def _update_tess(self):
        if self.T.updated:
            self._cache_to_clear_on_update.clear()
            self.T.updated = <cbool>False

    @staticmethod
    def _update_to_tess(func):
        def wrapped_func(solf, *args, **kwargs):
            solf._update_tess()
            return func(solf, *args, **kwargs)
        return wrapped_func

    @staticmethod
    def _dependent_property(fget):
        attr = '_'+fget.__name__
        def wrapped_fget(solf):
            solf._update_tess()
            if attr not in solf._cache_to_clear_on_update:
                solf._cache_to_clear_on_update[attr] = fget(solf)
            return solf._cache_to_clear_on_update[attr]
        return property(wrapped_fget, None, None, fget.__doc__)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.n = 0
        self.T = new Delaunay_with_info_2[uint32_t]()

    def is_valid(self):
        r"""Determine if the triangulation is a valid Delaunay triangulation.

        Returns:
            bool: True if the triangulation is valid, False otherwise.
        
        """

        return <pybool>self.T.is_valid()

    def write_to_file(self, fname):
        r"""Write the serialized tessellation information to a file.

        Args:
            fname (str): The full path to the file that the tessellation should 
                be written to.

        """
        cdef char* cfname = fname
        self.T.write_to_file(cfname)

    @_update_to_tess
    def read_from_file(self, fname):
        r"""Read serialized tessellation information from a file.

        Args:
            fname (str): The full path to the file that the tessellation should 
                be read from.

        """
        cdef char* cfname = fname
        self.T.read_from_file(cfname)
        self.n = self.num_finite_verts

    def plot(self, *args, **kwargs):
        r"""Plot the triangulation.

        Args:
            *args: All arguments are passed to :func:`plot.plot2D`.
            **kwargs: All keyword arguments are passed to :func:`plot.plot2D`.

        """
        plot.plot2D(self, *args, **kwargs)

    @_dependent_property
    def num_finite_verts(self): 
        r"""int: The number of finite vertices in the triangulation."""
        return self.T.num_finite_verts()
    @_dependent_property
    def num_finite_edges(self): 
        r"""int: The number of finite edges in the triangulation."""
        return self.T.num_finite_edges()
    @_dependent_property
    def num_finite_cells(self): 
        r"""int: The number of finite cells in the triangulation."""
        return self.T.num_finite_cells()
    @_dependent_property
    def num_infinite_verts(self):
        r"""int: The number of infinite vertices in the triangulation."""
        return self.T.num_infinite_verts()
    @_dependent_property
    def num_infinite_edges(self):
        r"""int: The number of infinite edges in the triangulation."""
        return self.T.num_infinite_edges()
    @_dependent_property
    def num_infinite_cells(self): 
        r"""int: The number of infinite cells in the triangulation."""
        return self.T.num_infinite_cells()
    @_dependent_property
    def num_verts(self): 
        r"""int: The total number of vertices (Finite + infinite) in the 
        triangulation."""
        return self.T.num_verts()
    @_dependent_property
    def num_edges(self): 
        r"""int: The total number of edges (finite + infinite) in the 
        triangulation."""
        return self.T.num_edges()
    @_dependent_property
    def num_cells(self): 
        r"""int: The total number of cells (finite + infinite) in the 
        triangulation."""
        return self.T.num_cells()

    @_update_to_tess
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

    @_update_to_tess
    def clear(self):
        r"""Removes all vertices and cells from the triangulation."""
        self.T.clear()


    @_dependent_property
    def vertices(self):
        r"""ndarray: The x,y coordinates of the vertices"""
        cdef np.ndarray[np.float64_t, ndim=2] out
        out = np.zeros([self.n, 2], 'float64')
        self.T.info_ordered_vertices(&out[0,0])
        return out

    @_dependent_property
    def edges(self):
        r""":obj:`ndarray` of uint64: Vertex index pairs for edges."""
        cdef np.ndarray[np.uint32_t, ndim=2] out
        out = np.zeros([self.num_finite_edges, 2], 'uint32')
        self.T.edge_info(&out[0,0])
        return out
        
    @_update_to_tess
    def remove(self, Delaunay2_vertex x):
        r"""Remove a vertex from the triangulation.

        Args:
            x (Delaunay2_vertex): Vertex that should be removed.

        """
        self.T.remove(x.x)

    @_update_to_tess
    def move(self, Delaunay2_vertex x, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location. If there is a vertex at the given 
        given coordinates, return that vertex and remove the one that was being 
        moved.

        Args:
            x (Delaunay2_vertex): Vertex that should be moved.
            pos (:obj:`ndarray` of float64): x,y coordinates that the vertex 
                be moved to.

        Returns:
            Delaunay2_vertex: Vertex at the new position.

        """
        assert(len(pos) == 2)
        cdef Delaunay_with_info_2[uint32_t].Vertex v
        v = self.T.move(x.x, &pos[0])
        cdef Delaunay2_vertex out = Delaunay2_vertex()
        out.assign(self.T, v)
        return out

    @_update_to_tess
    def move_if_no_collision(self, Delaunay2_vertex x, 
                             np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location only if there is not already a 
        vertex at the given coordinates. If there is a vertex there, it is 
        returned and the vertex being moved remains at its original location.

        Args:
            x (Delaunay2_vertex): Vertex that should be moved.
            pos (:obj:`ndarray` of float64): x,y coordinates that the vertex 
                be moved to.

        Returns:
            Delaunay2_vertex: Vertex at the new position.

        """
        assert(len(pos) == 2)
        cdef Delaunay_with_info_2[uint32_t].Vertex v
        v = self.T.move_if_no_collision(x.x, &pos[0])
        cdef Delaunay2_vertex out = Delaunay2_vertex()
        out.assign(self.T, v)
        return out

    @_update_to_tess
    def flip(self, Delaunay2_cell x, int i):
        r"""Flip the edge incident to cell x and neighbor i of cell x. The 
        method first checks if the edge can be flipped. (In the 2D case, it 
        can always be flipped).

        Args:
            x (Delaunay2_cell): Cell with edge that should be flipped.
            i (int): Integer specifying neighbor of x that is also incident 
                to the edge that should be flipped.

        Returns:
            bool: True if facet was flipped, False otherwise. (2D edges can 
                always be flipped).

        """
        return <pybool>self.T.flip(x.x, i)

    @_update_to_tess
    def flip_flippable(self, Delaunay2_cell x, int i):
        r"""Same as :meth:`Delaunay2.flip`, but assumes that facet is flippable 
        and does not check.

        Args:
            x (Delaunay2_cell): Cell with edge that should be flipped.
            i (int): Integer specifying neighbor of x that is also incident 
                to the edge that should be flipped.

        """
        self.T.flip_flippable(x.x, i)

    def get_vertex(self, np.uint64_t index):
        r"""Get the vertex object corresponding to the given index.

        Args:
            index (np.uint64_t): Index of vertex that should be found.

        Returns:
            Delaunay2_vertex: Vertex corresponding to the given index. If the 
                index is not found, the infinite vertex is returned.

        """
        cdef Delaunay_with_info_2[uint32_t].Vertex v = self.T.get_vertex(index)
        cdef Delaunay2_vertex out = Delaunay2_vertex()
        out.assign(self.T, v)
        return out

    def locate(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Get the cell/edge that a given point is a part of.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.

        Returns:
            :obj:`Delaunay2_cell` or :obj:`Delaunay2_edge`: Cell or edge that 
                the given point is a part of.

        """
        assert(len(pos) == 2)
        cdef int lt, li
        lt = li = 999
        cdef Delaunay2_cell c = Delaunay2_cell()
        c.assign(self.T, self.T.locate(&pos[0], lt, li))
        assert(lt != 999)
        if lt < 2:
            assert(li != 999)
        if lt == 0:
            return c.vertex(li)
        elif lt == 1:
            return Delaunay2_edge.from_cell(c, li)
        elif lt == 2:
            return c
        elif lt == 3:
            print("Point {} is outside the convex hull.".format(pos))
            return c
        elif lt == 4:
            print("Point {} is outside the affine hull.".format(pos))
            return 0
        else:
            raise RuntimeError("Value of {} not expected from CGAL locate.".format(lt))

    @property
    def all_verts_begin(self):
        r"""Delaunay2_vertex_iter: Starting vertex for all vertices in the 
        triangulation."""
        return Delaunay2_vertex_iter(self, 'all_begin')
    @property
    def all_verts_end(self):
        r"""Delaunay2_vertex_iter: Final vertex for all vertices in the 
        triangulation."""
        return Delaunay2_vertex_iter(self, 'all_end')
    @property
    def all_verts(self):
        r"""Delaunay2_vertex_range: Iterable for all vertices in the 
        triangulation."""
        return Delaunay2_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end)
    @property
    def finite_verts(self):
        r"""Delaunay2_vertex_range: Iterable for finite vertices in the 
        triangulation."""
        return Delaunay2_vertex_range(self.all_verts_begin, 
                                      self.all_verts_end, finite = True)

    @property
    def all_edges_begin(self):
        r"""Delaunay2_edge_iter: Starting edge for all edges in the 
        triangulation."""
        return Delaunay2_edge_iter(self, 'all_begin')
    @property
    def all_edges_end(self):
        r"""Delaunay2_edge_iter: Final edge for all edges in the 
        triangulation."""
        return Delaunay2_edge_iter(self, 'all_end')
    @property
    def all_edges(self):
        r"""Delaunay2_edge_range: Iterable for all edges in the 
        triangulation."""
        return Delaunay2_edge_range(self.all_edges_begin, 
                                    self.all_edges_end)
    @property
    def finite_edges(self):
        r"""Delaunay2_edge_range: Iterable for finite edges in the 
        triangulation."""
        return Delaunay2_edge_range(self.all_edges_begin, 
                                    self.all_edges_end, finite = True)

    @property
    def all_cells_begin(self):
        r"""Delaunay2_cell_iter: Starting cell for all cells in the triangulation."""
        return Delaunay2_cell_iter(self, 'all_begin')
    @property
    def all_cells_end(self):
        r"""Delaunay2_cell_iter: Final cell for all cells in the triangulation."""
        return Delaunay2_cell_iter(self, 'all_end')
    @property
    def all_cells(self):
        r"""Delaunay2_cell_range: Iterable for all cells in the
        triangulation."""
        return Delaunay2_cell_range(self.all_cells_begin,
                                    self.all_cells_end)
    @property
    def finite_cells(self):
        r"""Delaunay2_cell_range: Iterable for finite cells in the
        triangulation."""
        return Delaunay2_cell_range(self.all_cells_begin,
                                    self.all_cells_end, finite = True)

    def nearest_vertex(self, np.ndarray[np.float64_t, ndim=1] x):
        r"""Determine which vertex is closes to a given set of x,y coordinates

        Args:
            x (:obj:`ndarray` of float64): x,y coordinates.

        Returns:
            Delaunay2_vertex: Vertex closest to x.

        """
        cdef Delaunay_with_info_2[uint32_t].Vertex vc
        vc = self.T.nearest_vertex(&x[0])
        cdef Delaunay2_vertex v = Delaunay2_vertex()
        v.assign(self.T, vc)
        return v

    def get_boundary_of_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                                  Delaunay2_cell start):
        r"""Get the edges of the cell in conflict with a given point.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.
            start (Delaunay2_cell): Cell to start list of edges at.

        Returns:
            :obj:`list` of Delaunay2_edge: Edges of the cell in conflict with 
                 pos.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Edge] ev
        ev = self.T.get_boundary_of_conflicts(&pos[0], start.x)
        cdef object out = []
        cdef np.uint32_t i
        cdef Delaunay2_edge x
        for i in range(ev.size()):
            x = Delaunay2_edge()
            x.assign(self.T, ev[i])
            out.append(x)
        return out
        
    def get_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                      Delaunay2_cell start):
        r"""Get the cells that are in conflict with a given point.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates. 
            start (Delaunay2_cell): Cell to start list of conflicts at.

        Returns:
            :obj:`list` of Delaunay2_cell: Cells in conflict with pos.

        """
        cdef vector[Delaunay_with_info_2[uint32_t].Cell] cv
        cv = self.T.get_conflicts(&pos[0], start.x)
        cdef object out = []
        cdef np.uint32_t i
        cdef Delaunay2_cell x
        for i in range(cv.size()):
            x = Delaunay2_cell()
            x.assign(self.T, cv[i])
            out.append(x)
        return out

    def get_conflicts_and_boundary(self, np.ndarray[np.float64_t, ndim=1] pos,
                                   Delaunay2_cell start):
        r"""Get the cells and edges of cells that are in conflict with a given 
            point.

        Args:
            pos (:obj:`ndarray` of float64): x,y coordinates.
            start (Delaunay2_cell): Cell to start list of conflicts at.  
        
        Returns:
            tuple: :obj:`list` of :obj:`Delaunay2_cell`s in conflict with pos 
                and :obj:`list` of :obj:`Delaunay2_edge`s bounding the 
                conflicting cells.

        """
        cdef pair[vector[Delaunay_with_info_2[uint32_t].Cell],
                  vector[Delaunay_with_info_2[uint32_t].Edge]] cv
        cv = self.T.get_conflicts_and_boundary(&pos[0], start.x)
        cdef object out_cells = []
        cdef object out_edges = []
        cdef np.uint32_t i
        cdef Delaunay2_cell c
        cdef Delaunay2_edge e
        for i in range(cv.first.size()):
            c = Delaunay2_cell()
            c.assign(self.T, cv.first[i])
            out_cells.append(c)
        for i in range(cv.second.size()):
            e = Delaunay2_edge()
            e.assign(self.T, cv.second[i])
            out_edges.append(e)
        return out_cells, out_edges
