cimport numpy as np
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

cdef extern from "c_delaunayD.hpp":
    cdef int VALID
    cdef int D

    cdef cppclass Delaunay_with_info_D[Info] nogil:
        Delaunay_with_info_D() except +
        Delaunay_with_info_D(double *pts, Info *val, uint32_t n) except +
        bool updated
        bool is_valid() const
        uint32_t num_dims() const 
        uint32_t num_finite_verts() const
        uint32_t num_finite_cells() const
        uint32_t num_finite_faces(int d)
        uint32_t num_infinite_verts() const
        uint32_t num_infinite_cells() const
        uint32_t num_infinite_faces(int d)
        uint32_t num_verts() const
        uint32_t num_cells() const
        uint32_t num_faces(int d)

        bool is_equal(const Delaunay_with_info_D[Info] other) const

        cppclass Vertex
        cppclass Face
        cppclass Facet
        cppclass Cell

        Vertex infinite_vertex() const

        void insert(double *, Info *val, uint32_t n) except +
        void remove(Vertex) except +
        void clear() except + 

        # void write_to_file(const char* filename) except +
        # void read_from_file(const char* filename) except +
        I serialize[I](I &n, I &m, int32_t &d,
                       double* vert_pos, Info* vert_info,
                       I* cells, I* neighbors) const
        Info serialize_idxinfo[I](I &n, I &m, int32_t &d,
                                  Info* cells, I* neighbors) const
        I serialize_info2idx[I](I &n, I &m, int32_t &d,
                                I* faces, I* neighbors,
                                Info max_info, I* idx) const
        void deserialize[I](I n, I m, int32_t d,
                         double* vert_pos, Info* vert_info,
                         I* cells, I* neighbors, I idx_inf)
        void deserialize_idxinfo[I](I n, I m, int32_t d, double* vert_pos,
                                    I* cells, I* neighbors, I idx_inf)

        Vertex get_vertex(Info index) except +
        Cell locate(double* pos, int& lt, Face &f, Facet &ft)
        Cell locate(double* pos, int& lt, Face &f, Facet &ft, Cell c)

        void info_ordered_vertices(double* pos)
        void vertex_info(Info* verts)

        cppclass All_verts_iter:
            All_verts_iter()
            All_verts_iter& operator++()
            All_verts_iter& operator--()
            bool operator==(All_verts_iter other)
            bool operator!=(All_verts_iter other)
            Vertex vertex()
        All_verts_iter all_verts_begin()
        All_verts_iter all_verts_end()

        cppclass Vertex:
            Vertex()
            Vertex(All_verts_iter v)
            bool operator==(Vertex other)
            bool operator!=(Vertex other)
            void point(double* out)
            vector[double] point()
            Info info()
            Cell cell()
            void set_point(double* x)
            void set_cell(Cell c)

        cppclass Face:
            Face()
            # bool operator==(Face other)
            # bool operator!=(Face other)
            Vertex vertex(int i)
            int ind(int i)
            Cell cell()
            int dim()

        cppclass Facet:
            Facet()
            Facet(Cell x, int i1)
            # bool operator==(Facet other)
            # bool operator!=(Facet other)
            Cell cell()
            int ind()
            Vertex vertex(int i)

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
            void set_neighbor(int i, Cell c)

        bool are_equal(Face f1, Face f2)
        bool are_equal(Facet f1, Facet f2)

        bool is_infinite(Vertex x)
        bool is_infinite(Face x)
        bool is_infinite(Facet x)
        bool is_infinite(Cell x)
        bool is_infinite(All_verts_iter x)
        bool is_infinite(All_cells_iter x)

        vector[Vertex] incident_vertices(Vertex x)
        vector[Vertex] incident_vertices(Face x)
        vector[Vertex] incident_vertices(Cell x)

        vector[Face] incident_faces(Vertex x, int i)
        vector[Face] incident_faces(Face x, int i)
        vector[Face] incident_faces(Cell x, int i)

        vector[Cell] incident_cells(Vertex x)
        vector[Cell] incident_cells(Face x)
        vector[Cell] incident_cells(Cell x)

        int mirror_index(Cell x, int i) const
        Vertex mirror_vertex(Cell x, int i) const

        void circumcenter(Cell x, double* out) const
        double n_simplex_volume(Face f) const
        double n_simplex_volume(Facet f) const
        double dual_volume(const Vertex v)
        void dual_volumes(double* vols)

        vector[vector[Info]] outgoing_points(uint64_t nbox,
                                             double *left_edges, double *right_edges)
