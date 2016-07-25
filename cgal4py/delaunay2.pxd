# distutils: language = c++
# distutils: libraries = CGAL
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1

cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t

cdef extern from "c_delaunay2.hpp":
    cdef cppclass Delaunay_with_info_2[Info]:
        Delaunay_with_info_2() except +
        Delaunay_with_info_2(double *pts, Info *val, uint32_t n) except +
        uint32_t num_finite_verts()
        uint32_t num_finite_edges()
        uint32_t num_finite_cells()
        uint32_t num_infinite_verts()
        uint32_t num_infinite_edges()
        uint32_t num_infinite_cells()
        uint32_t num_verts()
        uint32_t num_edges()
        uint32_t num_cells()

        cppclass Vertex
        cppclass Edge
        cppclass Cell

        void insert(double *, Info *val, uint32_t n) except +
        void remove(Vertex v) except +
        void clear() except +
        Vertex move(Vertex v, double *pos) except +
        Vertex move_if_no_collision(Vertex v, double *pos) except +

        void write_to_file(const char* filename) except +
        void read_from_file(const char* filename) except +

        Vertex get_vertex(Info index)

        void info_ordered_vertices(double* pos)
        void vertex_info(Info* verts)
        void edge_info(Info* edges)

        cppclass All_verts_iter:
            All_verts_iter()
            All_verts_iter& operator++()
            All_verts_iter& operator--()
            bool operator==(All_verts_iter other)
            bool operator!=(All_verts_iter other)
        All_verts_iter all_verts_begin()
        All_verts_iter all_verts_end()

        cppclass Vertex:
            Vertex()
            Vertex(All_verts_iter v)
            bool operator==(Vertex other)
            bool operator!=(Vertex other)
            void point(double* out)
            Info info()
            Cell cell()
            void set_cell(Cell c)
            void set_point(double* x)

        cppclass All_edges_iter:
            All_edges_iter()
            All_edges_iter& operator++()
            All_edges_iter& operator--()
            bool operator==(All_edges_iter other)
            bool operator!=(All_edges_iter other)
        All_edges_iter all_edges_begin()
        All_edges_iter all_edges_end()

        cppclass Edge:
            Edge()
            Edge(All_edges_iter it)
            Edge(Vertex x1, Vertex x2)
            Edge(Cell x, int i)
            Vertex v1()
            Vertex v2()
            bool operator==(Edge other)
            bool operator!=(Edge other)

        cppclass All_cells_iter:
            All_cells_iter()
            All_cells_iter& operator++()
            All_cells_iter& operator--()
            bool operator==(All_cells_iter other)
            bool operator!=(All_cells_iter other)
        All_cells_iter all_cells_begin()
        All_cells_iter all_cells_end()

        cppclass Cell:
            Cell()
            Cell(All_cells_iter c)
            Cell(Vertex v1, Vertex v2, Vertex v3)
            Cell(Vertex v1, Vertex v2, Vertex v3, Cell c1, Cell c2, Cell c3)
            bool operator==(Cell other)
            bool operator!=(Cell other)

            Vertex vertex(int i)
            bool has_vertex(Vertex v)
            bool has_vertex(Vertex v, int *i)
            int ind(Vertex v)

            Cell neighbor(int i)
            bool has_neighbor(Cell c)
            bool has_neighbor(Cell c, int *i)
            int ind(Cell c)

            void set_vertex(int i, Vertex v)
            void set_vertices()
            void set_vertices(Vertex v1, Vertex v2, Vertex v3)
            void set_neighbor(int i, Cell c)
            void set_neighbors()
            void set_neighbors(Cell c1, Cell c2, Cell c3)

            void reorient()
            void ccw_permute()
            void cw_permute()

            int dimension() except + 

        bool is_infinite(Vertex x)
        bool is_infinite(Edge x)
        bool is_infinite(Cell x)
        bool is_infinite(All_verts_iter x)
        bool is_infinite(All_edges_iter x)
        bool is_infinite(All_cells_iter x)

        vector[Vertex] incident_vertices(Vertex x)
        vector[Edge] incident_edges(Vertex x)
        vector[Cell] incident_cells(Vertex x)

        vector[Vertex] incident_vertices(Edge x)
        vector[Edge] incident_edges(Edge x)
        vector[Cell] incident_cells(Edge x)

        vector[Vertex] incident_vertices(Cell x)
        vector[Edge] incident_edges(Cell x)
        vector[Cell] incident_cells(Cell x)

        Vertex nearest_vertex(double* pos)
        void circumcenter(Cell x, double* out)
        double dual_area(const Vertex v)
        double length(const Edge e)
        bool flip(Cell x, int i)
        void flip_flippable(Cell x, int i)

cdef class Delaunay2:
    cdef int n
    cdef Delaunay_with_info_2[uint32_t] *T
    cdef void _insert(self, np.ndarray[double, ndim=2, mode="c"] pts)

