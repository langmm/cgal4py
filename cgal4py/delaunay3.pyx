"""
delaunay3.pyx

Wrapper for CGAL 3D Delaunay Triangulation
"""

import cython

import numpy as np
cimport numpy as np

from . import plot

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

    def __repr__(self):
        return "Delaunay3_vertex[{} at {:+7.2e},{:+7.2e},{:+7.2e}]".format(
            self.index, *list(self.point))

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

    def set_point(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Set this vertex's corrdinates.

        Args:
            pos (:obj:`ndarray` of float64): new x,y,z coordinates for vertex.

        """
        assert(len(pos) == 3)
        self.x.set_point(&pos[0])

    def set_cell(self, Delaunay3_cell c):
        r"""Set the designated cell for this vertex.

        Args:
            c (Delaunay3_cell): Cell that should be set as the designated cell.

        """
        self.x.set_cell(c.x)

    property point:
        r""":obj:`ndarray` of :obj:`float64`: The cartesian (x,y,z) coordinates 
        of the vertex."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
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
        r"""float64: The volume of the dual Voronoi cell. If the volume is 
        infinite, -1.0 is returned."""
        def __get__(self):
            cdef np.float64_t out = self.T.dual_volume(self.x)
            return out

    property cell:
        r"""Delaunay3_cell: Designated cell for this vertex."""
        def __get__(self):
            cdef Delaunay_with_info_3[uint32_t].Cell c
            c = self.x.cell()
            cdef Delaunay3_cell out = Delaunay3_cell()
            out.assign(self.T, c)
            return out

    def incident_vertices(self):
        r"""Find vertices that are adjacent to this vertex.

        Returns:
            Delaunay3_vertex_vector: Iterator over vertices incident to this 
                vertex.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef Delaunay3_vertex_vector out = Delaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this vertex.

        Returns:
            Delaunay3_edge_vector: Iterator over edges incident to this vertex.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef Delaunay3_edge_vector out = Delaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this vertex.

        Returns:
            Delaunay3_facet_vector: Iterator over facets incident to this vertex.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef Delaunay3_facet_vector out = Delaunay3_facet_vector()
        out.assign(self.T, it)
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

    def __iter__(self):
        return self

    def __next__(self):
        cdef Delaunay3_vertex out
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        if self.x != self.xstop:
            out = self.x.vertex
            self.x.increment()
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

    def __getitem__(self, i):
        cdef Delaunay3_vertex out
        if isinstance(i, int):
            out = Delaunay3_vertex()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("Delaunay3_vertex_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class Delaunay3_edge:
    r"""Wrapper class for a triangulation edge.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this edge belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].Edge`): C++ edge object 
            Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].Edge x

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     Delaunay_with_info_3[uint32_t].Edge x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
                that this edge belongs to. 
            x (:obj:`Delaunay_with_info_3[uint32_t].Edge`): C++ edge object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    def __repr__(self):
        return "Delaunay3_edge[{},{}]".format(repr(self.vertex1),
                                              repr(self.vertex2))

    def __richcmp__(Delaunay3_edge self, Delaunay3_edge solf, int op):
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

    def is_Gabriel(self):
        r"""Determines if the edge is Gabriel (does not contain any other 
            vertices in it's smallest circumsphere).

        Returns:
            bool: True if the edge is Gabriel, False otherwise.

        """
        return <pybool>self.T.is_Gabriel(self.x)

    def is_equivalent(self, Delaunay3_edge solf):
        r"""Determine if another edge has the same vertices as this edge.

        Args:
            solf (Delaunay3_edge): Edge for comparison.

        Returns:
            bool: True if the two edges share the same vertices, False otherwise.

        """
        return <pybool>self.T.are_equal(self.x, solf.x)

    def vertex(self, int i):
        r"""Get the ith vertex on this edge.

        Args:
            i (int): Index of vertex to return.

        Returns:
            Delaunay3_vertex: ith vertex on this edge.

        """
        cdef Delaunay_with_info_3[uint32_t].Vertex x
        x = self.x.vertex(i)
        cdef Delaunay3_vertex out = Delaunay3_vertex()
        out.assign(self.T, x)
        return out

    property vertex1:
        r"""Delaunay3_vertex: The first vertex in the edge."""
        def __get__(self):
            cdef Delaunay_with_info_3[uint32_t].Vertex x = self.x.v1()
            cdef Delaunay3_vertex out = Delaunay3_vertex()
            out.assign(self.T, x)
            return out

    property vertex2:
        r"""Delaunay3_vertex: The second vertex in the edge."""
        def __get__(self):
            cdef Delaunay_with_info_3[uint32_t].Vertex x = self.x.v2()
            cdef Delaunay3_vertex out = Delaunay3_vertex()
            out.assign(self.T, x)
            return out

    property cell:
        r"""Delaunay3_cell: The cell this edge is assigned to."""
        def __get__(self):
            cdef Delaunay_with_info_3[uint32_t].Cell c
            c = self.x.cell()
            cdef Delaunay3_cell out = Delaunay3_cell()
            out.assign(self.T, c)
            return out

    property ind1:
        r"""int: The index of the 1st vertex of this edge in its cell."""
        def __get__(self):
            return self.x.ind1()

    property ind2:
        r"""int: The index of the 2nd vertex of this edge in its cell."""
        def __get__(self):
            return self.x.ind2()

    property length:
        r"""float64: The length of the edge. If infinite, -1 is returned"""
        def __get__(self):
            cdef np.float64_t out = self.T.length(self.x)
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this edge.

        Returns:
            Delaunay3_vertex_vector: Iterator over vertices incident to this 
                edge.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef Delaunay3_vertex_vector out = Delaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this edge.

        Returns:
            Delaunay3_edge_vector: Iterator over edges incident to this edge. 

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef Delaunay3_edge_vector out = Delaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this edge.

        Returns:
            Delaunay3_facet_vector: Iterator over facets incident to this edge. 

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef Delaunay3_facet_vector out = Delaunay3_facet_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this edge.

        Returns:
            Delaunay3_cell_vector: Iterator over cells incident to this edge.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef Delaunay3_cell_vector out = Delaunay3_cell_vector()
        out.assign(self.T, it)
        return out


cdef class Delaunay3_edge_iter:
    r"""Wrapper class for a triangulation edge iterator.

    Args:
        T (Delaunay3): Triangulation that this edge belongs to.
        edge (:obj:`str`, optional): String specifying the edge that 
            should be referenced. Valid options include: 
                'all_begin': The first edge in an iteration over all edges.
                'all_end': The last edge in an iteration over all edges.
 
    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this edge belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].All_edges_iter`): C++ edge  
            object. Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].All_edges_iter x

    def __cinit__(self, Delaunay3 T, str edge = None):
        self.T = T.T
        if edge == 'all_begin':
            self.x = self.T.all_edges_begin()
        elif edge == 'all_end':
            self.x = self.T.all_edges_end()

    def __richcmp__(Delaunay3_edge_iter self, Delaunay3_edge_iter solf, 
                    int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the edge is incident to the the infinite vertex.
        
        Returns:
            bool: True if the edge is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def increment(self):
        r"""Advance to the next edge in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous edge in the triangulation."""
        predecrement(self.x)

    property edge:
        r"""Delaunay3_edge: Corresponding edge object."""
        def __get__(self):
            cdef Delaunay3_edge out = Delaunay3_edge()
            out.assign(self.T, Delaunay_with_info_3[uint32_t].Edge(self.x))
            return out


cdef class Delaunay3_edge_range:
    r"""Wrapper class for iterating over a range of triangulation edges.

    Args:
        xstart (Delaunay3_edge_iter): The starting edge.  
        xstop (Delaunay3_edge_iter): Final edge that will end the iteration. 
        finite (:obj:`bool`, optional): If True, only finite edges are 
            iterated over. Otherwise, all edges are iterated over. Defaults 
            False.

    Attributes:
        x (Delaunay3_edge_iter): The currentedge. 
        xstop (Delaunay3_edge_iter): Final edge that will end the iteration. 
        finite (bool): If True, only finite edges are iterater over. Otherwise
            all edges are iterated over. 

    """
    cdef Delaunay3_edge_iter x
    cdef Delaunay3_edge_iter xstop
    cdef pybool finite
    def __cinit__(self, Delaunay3_edge_iter xstart, 
                  Delaunay3_edge_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef Delaunay3_edge out
        if self.x != self.xstop:
            out = self.x.edge
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class Delaunay3_edge_vector:
    r"""Wrapper class for a vector of edges.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_3[uint32_t].Edge]`): Vector of C++ 
            edges.
        n (int): The number of edges in the vector.
        i (int): The index of the currect edge.

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef vector[Delaunay_with_info_3[uint32_t].Edge] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     vector[Delaunay_with_info_3[uint32_t].Edge] v):
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
        cdef Delaunay3_edge out
        if self.i < self.n:
            out = Delaunay3_edge()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef Delaunay3_edge out
        if isinstance(i, int):
            out = Delaunay3_edge()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("Delaunay3_edge_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class Delaunay3_facet:
    r"""Wrapper class for a triangulation facet.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this facet belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].Facet`): C++ facet object 
            Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].Facet x

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     Delaunay_with_info_3[uint32_t].Facet x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
                that this facet belongs to. 
            x (:obj:`Delaunay_with_info_3[uint32_t].Facet`): C++ facet object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    def __repr__(self):
        return "Delaunay3_facet[{},{},{}]".format(repr(self.vertex(0)),
                                                  repr(self.vertex(1)),
                                                  repr(self.vertex(2)))

    def __richcmp__(Delaunay3_facet self, Delaunay3_facet solf, int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the facet is incident to the infinite vertex.
        
        Returns:
            bool: True if the facet is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def is_Gabriel(self):
        r"""Determines if the facet is Gabriel (does not contain any other 
            vertices in it's smallest circumsphere).

        Returns:
            bool: True if the facet is Gabriel, False otherwise.

        """
        return <pybool>self.T.is_Gabriel(self.x)

    def is_equivalent(self, Delaunay3_facet solf):
        r"""Determine if another facet has the same vertices as this facet.

        Args:
            solf (Delaunay3_facet): Facet for comparison.

        Returns:
            bool: True if the two facets share the same vertices, False 
                otherwise.

        """
        return <pybool>self.T.are_equal(self.x, solf.x)

    def vertex(self, int i):
        r"""Get the ith vertex incident to this facet.

        Args:
            i (int): Index of vertex that should be returned.

        Returns:
            Delaunay_vertex: ith vertex of this facet.

        """
        cdef Delaunay_with_info_3[uint32_t].Vertex v
        v = self.x.vertex(i)
        cdef Delaunay3_vertex out = Delaunay3_vertex()
        out.assign(self.T, v)
        return out

    property area:
        r"""float64: The area of the facet. If infinite, -1 is returned"""
        def __get__(self):
            return -1

    property cell:
        r"""Delaunay3_cell: The cell this facet is assigned to."""
        def __get__(self):
            cdef Delaunay_with_info_3[uint32_t].Cell c
            c = self.x.cell()
            cdef Delaunay3_cell out = Delaunay3_cell()
            out.assign(self.T, c)
            return out

    property ind:
        r"""int: The index of the vertex this facet is opposite on its cell."""
        def __get__(self):
            return self.x.ind()
        
    def incident_vertices(self):
        r"""Find vertices that are incident to this facet.

        Returns:
            Delaunay3_vertex_vector: Iterator over vertices incident to this 
                facet.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef Delaunay3_vertex_vector out = Delaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this facet.

        Returns:
            Delaunay3_edge_vector: Iterator over edges incident to this facet. 

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef Delaunay3_edge_vector out = Delaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this facet.

        Returns:
            Delaunay3_facet_vector: Iterator over facets incident to this facet. 

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef Delaunay3_facet_vector out = Delaunay3_facet_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this facet.

        Returns:
            Delaunay3_cell_vector: Iterator over cells incident to this facet.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef Delaunay3_cell_vector out = Delaunay3_cell_vector()
        out.assign(self.T, it)
        return out

    # Currently segfaults inside CGAL function
    # def side_of_circle(self, np.ndarray[np.float64_t, ndim=1] pos):
    #     r"""Determine where a point is with repect to this facet's 
    #         circumcircle. 
    #
    #     Args: 
    #         pos (:obj:`ndarray` of np.float64): x,y,z coordinates. 
    # 
    #     Returns: 
    #         int: -1, 0, or 1 if `pos` is within, on, or inside this facet's 
    #             circumcircle respectively. 
    # 
    #     """
    #     return self.T.side_of_circle(self.x, &pos[0])


cdef class Delaunay3_facet_iter:
    r"""Wrapper class for a triangulation facet iterator.

    Args:
        T (Delaunay3): Triangulation that this facet belongs to.
        facet (:obj:`str`, optional): String specifying the facet that 
            should be referenced. Valid options include: 
                'all_begin': The first facet in an iteration over all facets.
                'all_end': The last facet in an iteration over all facets.
 
    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
            that this facet belongs to. 
        x (:obj:`Delaunay_with_info_3[uint32_t].All_facets_iter`): C++ facet  
            object. Direct interaction with this object is not recommended. 

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef Delaunay_with_info_3[uint32_t].All_facets_iter x

    def __cinit__(self, Delaunay3 T, str facet = None):
        self.T = T.T
        if facet == 'all_begin':
            self.x = self.T.all_facets_begin()
        elif facet == 'all_end':
            self.x = self.T.all_facets_end()

    def __richcmp__(Delaunay3_facet_iter self, Delaunay3_facet_iter solf, 
                    int op):
        if (op == 2):
            return <pybool>(self.x == solf.x)
        elif (op == 3):
            return <pybool>(self.x != solf.x)
        else:
            raise NotImplementedError

    def is_infinite(self):
        r"""Determine if the facet is incident to the the infinite vertex.
        
        Returns:
            bool: True if the facet is incident to the infinite vertex, False 
                otherwise.

        """
        return self.T.is_infinite(self.x)

    def increment(self):
        r"""Advance to the next facet in the triangulation."""
        preincrement(self.x)

    def decrement(self):
        r"""Advance to the previous facet in the triangulation."""
        predecrement(self.x)

    property facet:
        r"""Delaunay3_facet: Corresponding facet object."""
        def __get__(self):
            cdef Delaunay3_facet out = Delaunay3_facet()
            out.assign(self.T, Delaunay_with_info_3[uint32_t].Facet(self.x))
            return out


cdef class Delaunay3_facet_range:
    r"""Wrapper class for iterating over a range of triangulation facets.

    Args:
        xstart (Delaunay3_facet_iter): The starting facet.  
        xstop (Delaunay3_facet_iter): Final facet that will end the iteration. 
        finite (:obj:`bool`, optional): If True, only finite facets are 
            iterated over. Otherwise, all facets are iterated over. Defaults 
            False.

    Attributes:
        x (Delaunay3_facet_iter): The currentfacet. 
        xstop (Delaunay3_facet_iter): Final facet that will end the iteration. 
        finite (bool): If True, only finite facets are iterater over. Otherwise
            all facets are iterated over. 

    """
    cdef Delaunay3_facet_iter x
    cdef Delaunay3_facet_iter xstop
    cdef pybool finite
    def __cinit__(self, Delaunay3_facet_iter xstart, 
                  Delaunay3_facet_iter xstop,
                  pybool finite = False):
        self.x = xstart
        self.xstop = xstop
        self.finite = finite

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef Delaunay3_facet out
        if self.x != self.xstop:
            out = self.x.facet
            self.x.increment()
            return out
        else:
            raise StopIteration()

cdef class Delaunay3_facet_vector:
    r"""Wrapper class for a vector of facets.

    Attributes:
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object.
            Direct interaction with this object is not recommended.
        v (:obj:`vector[Delaunay_with_info_3[uint32_t].Facet]`): Vector of C++ 
            facets.
        n (int): The number of facets in the vector.
        i (int): The index of the currect facet.

    """
    cdef Delaunay_with_info_3[uint32_t] *T
    cdef vector[Delaunay_with_info_3[uint32_t].Facet] v
    cdef int n
    cdef int i

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     vector[Delaunay_with_info_3[uint32_t].Facet] v):
        r"""Assign C++ attributes.

        Args:
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object.
                Direct interaction with this object is not recommended.
            v (:obj:`vector[Delaunay_with_info_3[uint32_t].Facet]`): Vector of 
                C++ facets.

        """
        self.T = T
        self.v = v
        self.n = v.size()
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef Delaunay3_facet out
        if self.i < self.n:
            out = Delaunay3_facet()
            out.assign(self.T, self.v[self.i])
            self.i += 1
            return out
        else:
            raise StopIteration()

    def __getitem__(self, i):
        cdef Delaunay3_facet out
        if isinstance(i, int):
            out = Delaunay3_facet()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("Delaunay3_facet_vector indices must be itegers, "+
                            "not {}".format(type(i)))


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

    cdef void assign(self, Delaunay_with_info_3[uint32_t] *T,
                     Delaunay_with_info_3[uint32_t].Cell x):
        r"""Assign C++ objects to attributes.

            Args:
            T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ Triangulation object 
                that this cell belongs to. 
            x (:obj:`Delaunay_with_info_3[uint32_t].Cell`): C++ cell object 
                Direct interaction with this object is not recommended. 

        """
        self.T = T
        self.x = x

    def __repr__(self):
        return "Delaunay2_cell[{},{},{},{}]".format(repr(self.vertex(0)),
                                                    repr(self.vertex(1)),
                                                    repr(self.vertex(2)),
                                                    repr(self.vertex(3)))

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

    def vertex(self, int i):
        r"""Find the ith vertex that is incident to this cell. 

        Args:
            i (int): The index of the vertex that should be returned.

        Returns:
            Delaunay3_vertex: The ith vertex incident to this cell. 

        """
        cdef Delaunay_with_info_3[uint32_t].Vertex v
        v = self.x.vertex(i)
        cdef Delaunay3_vertex out = Delaunay3_vertex()
        out.assign(self.T, v)
        return out

    def has_vertex(self, Delaunay3_vertex v, pybool return_index = False):
        r"""Determine if a vertex belongs to this cell.

        Args:
            v (Delaunay3_vertex): Vertex to test ownership for. 
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
            
    def ind_vertex(self, Delaunay3_vertex v):
        r"""Determine the index of a vertex within a cell. 

        Args: 
            v (Delaunay3_vertex): Vertex to find index for. 

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
        cdef Delaunay_with_info_3[uint32_t].Cell v
        v = self.x.neighbor(i)
        cdef Delaunay3_cell out = Delaunay3_cell()
        out.assign(self.T, v)
        return out

    def has_neighbor(self, Delaunay3_cell v, pybool return_index = False):
        r"""Determine if a cell is a neighbor to this cell. 

        Args: 
            v (Delaunay3_cell): Cell to test as a neighbor. 
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

    def ind_neighbor(self, Delaunay3_cell v):
        r"""Determine the index of a neighboring cell. 

        Args: 
            v (Delaunay3_cell): Neighboring cell to find index for. 

        Returns: 
            int: Index of vertex opposite to neighboring cell. 

        """
        return self.x.ind(v.x)

    def set_vertex(self, int i, Delaunay3_vertex v):
        r"""Set the ith vertex of this cell. 

        Args: 
            i (int): Index of this cell's vertex that should be set. 
            v (Delauany3_vertex): Vertex to set ith vertex of this cell to. 

        """
        self.x.set_vertex(i, v.x)

    def set_vertices(self, Delaunay3_vertex v1, Delaunay3_vertex v2,
                     Delaunay3_vertex v3, Delaunay3_vertex v4):
        r"""Set this cell's vertices. 

        Args: 
            v1 (Delaunay2_vertex): 1st vertex of cell. 
            v2 (Delaunay2_vertex): 2nd vertex of cell. 
            v3 (Delaunay2_vertex): 3rd vertex of cell. 
            v4 (Delaunay2_vertex): 4th vertex of cell. 

        """
        self.x.set_vertices(v1.x, v2.x, v3.x, v4.x)

    def reset_vertices(self):
        r"""Reset all of this cell's vertices."""
        self.x.set_vertices()

    def set_neighbor(self, int i, Delaunay3_cell n):
        r"""Set the ith neighboring cell of this cell. 

        Args: 
            i (int): Index of this cell's neighbor that should be set. 
            n (Delaunay3_cell): Cell to set ith neighbor of this cell to. 

        """
        self.x.set_neighbor(i, n.x)

    def set_neighbors(self, Delaunay3_cell c1, Delaunay3_cell c2,
                      Delaunay3_cell c3, Delaunay3_cell c4):
        r"""Set this cell's neighboring cells. 

        Args: 
            c1 (Delaunay3_cell): 1st neighboring cell. 
            c2 (Delaunay3_cell): 2nd neighboring cell. 
            c3 (Delaunay3_cell): 3rd neighboring cell. 
            c4 (Delaunay3_cell): 4th neighboring cell. 

        """
        self.x.set_neighbors(c1.x, c2.x, c3.x, c4.x)

    def reset_neighbors(self):
        r"""Reset all of this cell's neighboring cells."""
        self.x.set_neighbors()

    property circumcenter:
        """:obj:`ndarray` of float64: x,y,z coordinates of cell circumcenter."""
        def __get__(self):
            cdef np.ndarray[np.float64_t] out = np.zeros(3, 'float64')
            self.T.circumcenter(self.x, &out[0])
            return out

    def incident_vertices(self):
        r"""Find vertices that are incident to this cell.

        Returns:
            Delaunay3_vertex_vector: Iterator over vertices incident to this 
                cell.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Vertex] it
        it = self.T.incident_vertices(self.x)
        cdef Delaunay3_vertex_vector out = Delaunay3_vertex_vector()
        out.assign(self.T, it)
        return out

    def incident_edges(self):
        r"""Find edges that are incident to this cell.

        Returns:
            Delaunay3_edge_vector: Iterator over edges incident to this cell. 

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Edge] it
        it = self.T.incident_edges(self.x)
        cdef Delaunay3_edge_vector out = Delaunay3_edge_vector()
        out.assign(self.T, it)
        return out

    def incident_facets(self):
        r"""Find facets that are incident to this cell.

        Returns:
            Delaunay3_facet_vector: Iterator over facets incident to this cell. 

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Facet] it
        it = self.T.incident_facets(self.x)
        cdef Delaunay3_facet_vector out = Delaunay3_facet_vector()
        out.assign(self.T, it)
        return out

    def incident_cells(self):
        r"""Find cells that are incident to this cell.

        Returns:
            Delaunay3_cell_vector: Iterator over cells incident to this cell.

        """
        cdef vector[Delaunay_with_info_3[uint32_t].Cell] it
        it = self.T.incident_cells(self.x)
        cdef Delaunay3_cell_vector out = Delaunay3_cell_vector()
        out.assign(self.T, it)
        return out

    def side_of_sphere(self, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Determine where a point is with repect to this cell's 
            circumsphere. 

        Args: 
            pos (:obj:`ndarray` of np.float64): x,y,z coordinates. 

        Returns: 
            int: -1, 0, or 1 if `pos` is within, on, or inside this cell's 
                circumsphere respectively. 

        """
        return self.T.side_of_sphere(self.x, &pos[0])


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

    def __iter__(self):
        return self

    def __next__(self):
        if self.finite:
            while (self.x != self.xstop) and self.x.is_infinite():
                self.x.increment()
        cdef Delaunay3_cell out
        if self.x != self.xstop:
            out = self.x.cell
            self.x.increment()
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

    def __getitem__(self, i):
        cdef Delaunay3_cell out
        if isinstance(i, int):
            out = Delaunay3_cell()
            out.assign(self.T, self.v[i])
            return out
        else:
            raise TypeError("Delaunay3_cell_vector indices must be itegers, "+
                            "not {}".format(type(i)))


cdef class Delaunay3:
    r"""Wrapper class for a 3D Delaunay triangulation.

    Attributes:
        n (int): The number of points inserted into the triangulation.
        T (:obj:`Delaunay_with_info_3[uint32_t]`): C++ triangulation object. 
            Direct interaction with this object is not recommended. 

    """

    _props_to_clear_on_update = {}

    @staticmethod
    def _dependent_property(fget):
        attr = '_'+fget.__name__
        def wrapped_fget(solf):
            if attr not in solf._props_to_clear_on_update:
                solf._props_to_clear_on_update[attr] = fget(solf)
            return solf._props_to_clear_on_update[attr]
        return property(wrapped_fget, None, None, fget.__doc__)

    @staticmethod
    def _update_to_tess(func):
        def wrapped_func(solf, *args, **kwargs):
            solf._props_to_clear_on_update.clear()
            return func(solf, *args, **kwargs)
        return wrapped_func

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.n = 0
        self.T = new Delaunay_with_info_3[uint32_t]()

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
            *args: All arguments are passed to :func:`plot.plot3D`.
            **kwargs: All keyword arguments are passed to :func:`plot.plot3D`.

        """
        plot.plot3D(self, *args, **kwargs)

    @_dependent_property
    def num_finite_verts(self): 
        r"""int: The number of finite vertices in the triangulation."""
        return self.T.num_finite_verts()
    @_dependent_property
    def num_finite_edges(self):
        r"""int: The number of finite edges in the triangulation."""
        return self.T.num_finite_edges()
    @_dependent_property
    def num_finite_facets(self):
        r"""int: The number of finite facets in the triangulation."""
        return self.T.num_finite_facets()
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
    def num_infinite_facets(self):
        r"""int: The number of infinite facets in the triangulation."""
        return self.T.num_infinite_facets()
    @_dependent_property
    def num_infinite_cells(self):
        r"""int: The number of infinite cells in the triangulation."""
        return self.T.num_infinite_cells()
    @_dependent_property
    def num_verts(self): 
        r"""int: The total number of vertices (finite + infinite) in the 
        triangulation."""
        return self.T.num_verts()
    @_dependent_property
    def num_edges(self):
        r"""int: The total number of edges (finite + infinite) in the 
        triangulation."""
        return self.T.num_edges()
    @_dependent_property
    def num_facets(self):
        r"""int: The total number of facets (finite + infinite) in the 
        triangulation."""
        return self.T.num_facets()
    @_dependent_property
    def num_cells(self):
        r"""int: The total number of cells (finite + infinite) in the 
        triangulation."""
        return self.T.num_cells()

    @_update_to_tess
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

    @_update_to_tess
    def clear(self):
        r"""Removes all vertices and cells from the triangulation."""
        self.T.clear()

    @_dependent_property
    def vertices(self):
        r"""ndarray: The x,y,z coordinates of the vertices"""
        cdef np.ndarray[np.float64_t, ndim=2] out
        out = np.zeros([self.n, 3], 'float64')
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
    def remove(self, Delaunay3_vertex x):
        r"""Remove a vertex from the triangulation. 

        Args: 
            x (Delaunay3_vertex): Vertex that should be removed. 

        """
        self.T.remove(x.x)

    @_update_to_tess
    def move(self, Delaunay3_vertex x, np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location. If there is a vertex at the given 
        given coordinates, return that vertex and remove the one that was being 
        moved. 

        Args: 
            x (Delaunay3_vertex): Vertex that should be moved. 
            pos (:obj:`ndarray` of float64): x,y,z coordinates that the vertex 
                be moved to. 

        Returns: 
            Delaunay3_vertex: Vertex at the new position. 

        """
        assert(len(pos) == 3)
        cdef Delaunay_with_info_3[uint32_t].Vertex v
        v = self.T.move(x.x, &pos[0])
        cdef Delaunay3_vertex out = Delaunay3_vertex()
        out.assign(self.T, v)
        return out

    @_update_to_tess
    def move_if_no_collision(self, Delaunay3_vertex x,
                             np.ndarray[np.float64_t, ndim=1] pos):
        r"""Move a vertex to a new location only if there is not already a 
        vertex at the given coordinates. If there is a vertex there, it is 
        returned and the vertex being moved remains at its original location. 

        Args: 
            x (Delaunay3_vertex): Vertex that should be moved. 
            pos (:obj:`ndarray` of float64): x,y,z coordinates that the vertex 
                be moved to. 

        Returns: 
            Delaunay2_vertex: Vertex at the new position. 

        """
        assert(len(pos) == 3)
        cdef Delaunay_with_info_3[uint32_t].Vertex v
        v = self.T.move_if_no_collision(x.x, &pos[0])
        cdef Delaunay3_vertex out = Delaunay3_vertex()
        out.assign(self.T, v)
        return out

    @_update_to_tess
    def flip(self, Delaunay3_cell x, int i):
        r"""Flip the facet incident to cell x and neighbor i of cell x. This 
        method first checks if the facet can be flipped.

        Args:
            x (Delaunay3_cell): Cell with facet that should be flipped.
            i (int): Integer specifying neighbor that is also incident to the 
                cell that should be flipped.

        Returns:
            bool: True if facet was flipped, False otherwise.

        """
        return <pybool>self.T.flip(x.x, i)

    @_update_to_tess
    def flip_flippable(self, Delaunay3_cell x, int i):
        r"""Same as :meth:`Delaunay3.flip`, but assumes that facet is flippable 
        and does not check.

        Args:
            x (Delaunay3_cell): Cell with facet that should be flipped.
            i (int): Integer specifying neighbor that is also incident to the 
                cell that should be flipped.
        """
        self.T.flip_flippable(x.x, i)

    def get_vertex(self, np.uint64_t index):
        r"""Get the vertex object corresponding to the given index. 

        Args: 
            index (np.uint64_t): Index of vertex that should be found. 

        Returns: 
            Delaunay3_vertex: Vertex corresponding to the given index. If the 
                index is not found, the infinite vertex is returned. 

        """
        cdef Delaunay_with_info_3[uint32_t].Vertex v = self.T.get_vertex(index)
        cdef Delaunay3_vertex out = Delaunay3_vertex()
        out.assign(self.T, v)
        return out

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
    def all_edges_begin(self):
        r"""Delaunay3_edge_iter: Starting edge for all edges in the 
        triangulation."""
        return Delaunay3_edge_iter(self, 'all_begin')
    @property
    def all_edges_end(self):
        r"""Delaunay3_edge_iter: Final edge for all edges in the 
        triangulation."""
        return Delaunay3_edge_iter(self, 'all_end')
    @property
    def all_edges(self):
        r"""Delaunay3_edge_range: Iterable for all edges in the 
        triangulation."""
        return Delaunay3_edge_range(self.all_edges_begin,
                                    self.all_edges_end)
    @property
    def finite_edges(self):
        r"""Delaunay3_edge_range: Iterable for finite edges in the 
        triangulation."""
        return Delaunay3_edge_range(self.all_edges_begin,
                                    self.all_edges_end, finite = True)

    @property
    def all_facets_begin(self):
        r"""Delaunay3_facet_iter: Starting facet for all facets in the 
        triangulation."""
        return Delaunay3_facet_iter(self, 'all_begin')
    @property
    def all_facets_end(self):
        r"""Delaunay3_facet_iter: Final facet for all facets in the 
        triangulation."""
        return Delaunay3_facet_iter(self, 'all_end')
    @property
    def all_facets(self):
        r"""Delaunay3_facet_range: Iterable for all facets in the 
        triangulation."""
        return Delaunay3_facet_range(self.all_facets_begin,
                                     self.all_facets_end)
    @property
    def finite_facets(self):
        r"""Delaunay3_facet_range: Iterable for finite facets in the 
        triangulation."""
        return Delaunay3_facet_range(self.all_facets_begin,
                                     self.all_facets_end, finite = True)

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

    def get_boundary_of_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                                  Delaunay3_cell start):
        r"""Get the facets bounding the zone in conflict with a given point. 

        Args: 
            pos (:obj:`ndarray` of float64): x,y,z coordinates. 
            start (Delaunay3_cell): Cell to start list of facets at. 

        Returns: 
            :obj:`list` of Delaunay3_facet: Facets bounding the zone in conflict 
                 with pos. 

        """
        cdef pair[vector[Delaunay_with_info_3[uint32_t].Cell],
                  vector[Delaunay_with_info_3[uint32_t].Facet]] cv
        cv = self.T.find_conflicts(&pos[0], start.x)
        cdef object out_facets = []
        cdef np.uint32_t i
        cdef Delaunay3_facet f
        for i in range(cv.second.size()):
            f = Delaunay3_facet()
            f.assign(self.T, cv.second[i])
            out_facets.append(f)
        return out_facets

    def get_conflicts(self, np.ndarray[np.float64_t, ndim=1] pos,
                      Delaunay3_cell start):
        r"""Get the cells in conflict with a given point.

        Args: 
            pos (:obj:`ndarray` of float64): x,y,z coordinates. 
            start (Delaunay3_cell): Cell to start list of facets at. 

        Returns: 
            :obj:`list` of Delaunay3_cell: Cells in conflict with pos. 

        """
        cdef pair[vector[Delaunay_with_info_3[uint32_t].Cell],
                  vector[Delaunay_with_info_3[uint32_t].Facet]] cv
        cv = self.T.find_conflicts(&pos[0], start.x)
        cdef object out_cells = []
        cdef np.uint32_t i
        cdef Delaunay3_cell c
        for i in range(cv.first.size()):
            c = Delaunay3_cell()
            c.assign(self.T, cv.first[i])
            out_cells.append(c)
        return out_cells

    def get_conflicts_and_boundary(self, np.ndarray[np.float64_t, ndim=1] pos,
                                   Delaunay3_cell start):
        r"""Get the cells in conflict with a given point and the facets bounding 
            the zone in conflict.

        Args: 
            pos (:obj:`ndarray` of float64): x,y,z coordinates. 
            start (Delaunay3_cell): Cell to start list of facets at. 

        Returns: 
            tuple: :obj:`list` of `Delaunay3_cell`s in conflict with pos and 
                :obj:`list` of `Delaunay3_facet`s bounding the zone in conflict.

        """
        cdef pair[vector[Delaunay_with_info_3[uint32_t].Cell],
                  vector[Delaunay_with_info_3[uint32_t].Facet]] cv
        cv = self.T.find_conflicts(&pos[0], start.x)
        cdef object out_facets = []
        cdef object out_cells = []
        cdef np.uint32_t i
        cdef Delaunay3_cell c
        cdef Delaunay3_facet f
        for i in range(cv.first.size()):
            c = Delaunay3_cell()
            c.assign(self.T, cv.first[i])
            out_cells.append(c)
        for i in range(cv.second.size()):
            f = Delaunay3_facet()
            f.assign(self.T, cv.second[i])
            out_facets.append(f)
        return out_cells, out_facets
