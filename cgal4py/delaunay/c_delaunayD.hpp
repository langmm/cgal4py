#include <vector>
#include <set>
#include <array>
#include <utility>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdint.h>
#ifdef READTHEDOCS
#define VALID 1
#include "dummy_CGAL.hpp"
#else
#define VALID 1
#include <CGAL/config.h>
#if (CGAL_VERSION_NR >= 1040601000)
#include <CGAL/Epick_d.h>
#include <CGAL/Delaunay_triangulation.h>
// #include <CGAL/Triangulation_full_cell.h>
// #include <CGAL/Triangulation_vertex.h>
// #include <CGAL/squared_distance_3.h>
#include <CGAL/Unique_hash_map.h>
#else
#include "dummy_CGAL.hpp"
#endif
#endif

template <uint32_t D, typename Info_>
class Delaunay_with_info_D
{
public:
  typedef CGAL::Epick_d< CGAL::Dimension_tag<D> >      K;
  typedef CGAL::Delaunay_triangulation<K>              Delaunay;
  typedef typename Delaunay::Point                     Point;
  // typedef typename Delaunay::Vertex                    Vertex_T;
  typedef typename Delaunay::Facet                     Facet_handle;
  // typedef typename Delaunay::Full_cell                 Cell_T;
  typedef typename Delaunay::Face                      Face_handle;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Full_cell_handle          Cell_handle;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Facet_iterator            Facet_iterator;
  typedef typename Delaunay::Full_cell_iterator        Cell_iterator;
  typedef typename Delaunay::Finite_vertex_iterator    Finite_vertex_iterator;
  typedef typename Delaunay::Finite_facet_iterator     Finite_facet_iterator;
  typedef typename Delaunay::Finite_full_cell_iterator Finite_cell_iterator;
  typedef typename Delaunay::Locate_type               Locate_type;
  typedef typename K::Cartesian_const_iterator_d       Cartesian_const_iterator_d;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>    Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Full_cell_handle,int> Cell_hash;
  typedef Info_ Info;
  uint32_t ndim = D;
  Delaunay T(D);
  bool updated = false;
  Delaunay_with_info_D() {};
  Delaunay_with_info_D(double *pts, Info *val, uint32_t n) {
    insert(pts, val, n);
  }
  bool is_valid() const { return T.is_valid(); }
  uint32_t num_finite_verts() const { return (uint32_t)(T.number_of_vertices()); }
  uint32_t num_finite_cells() const { return (uint32_t)(T.number_of_finite_full_cells()); }
  uint32_t num_infinite_verts() const { return 1; }
  uint32_t num_infinite_cells() const { return (num_cells() - num_finite_cells()); }
  uint32_t num_verts() const { return (num_finite_verts() + num_infinite_verts()); }
  uint32_t num_cells() const { return (uint32_t)(T.number_of_full_cells()); }
  bool is_equal(const Delaunay_with_info_3<Info> other) const {
    // Verts
    if (num_verts() != other.num_verts()) return false;
    if (num_finite_verts() != other.num_finite_verts()) return false;
    if (num_infinite_verts() != other.num_infinite_verts()) return false;
    // Cells
    if (num_cells() != other.num_cells()) return false;
    if (num_finite_cells() != other.num_finite_cells()) return false;
    if (num_infinite_cells() != other.num_infinite_cells()) return false;
    return true;
  }

  class Vertex;
  class Facet;
  class Cell;
  class Face;

  Point pos2point(double* pos) {
    std::vector<Info> vp;
    for (uint32_t i; i < ndim; i++)
      vp.push_back(pos[i]);
    return Point(vp.begin(), vp.end());
  }
  void insert(double *pts, Info *val, uint32_t n)
  {
    updated = true;
    uint32_t i, j;
    Vertex_handle v;
    std::vector<Info> vp;
    for (i = 0; i < n; i++) {
      v = T.insert(pos2point(pts+(ndim*i)));
      v->data() = val[i];
    }
  }
  void remove(Vertex v) { updated = true; T.remove(v._x); }
  void clear() { updated = true; T.clear(); }

  Vertex get_vertex(Info index) const {
    Finite_vertex_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      if (it->data() == index)
        return Vertex(static_cast<Vertex_handle>(it));
    }
    return Vertex(T.infinite_vertex());
  }

  Face locate(double* pos, int& lt, Face f, Facet ft) const {
    Point p = Point(pos[0], pos[1], pos[2]);
    Locate_type lt_out = Locate_type(0);
    Cell out = Cell(T.locate(p, lt_out, f._x, ft._x));
    lt = (int)lt_out;
    return out;
  }
  Face locate(double* pos, int& lt, Face f, Facet ft, Cell c) const {
    Point p = Point(pos[0], pos[1], pos[2]);
    Locate_type lt_out = Locate_type(0);
    Cell out = Cell(T.locate(p, lt_out, f._x, ft._x, c._x));
    lt = (int)lt_out;
    return out;
  }

  template <typename Wrap, typename Wrap_handle>
  class wrap_insert_iterator
  {
  protected:
    std::vector<Wrap>* container;

  public:
    explicit wrap_insert_iterator (std::vector<Wrap>& x) : container(&x) {}
    wrap_insert_iterator& operator= (Wrap_handle value) {
      Wrap value_wrap = Wrap();
      value_wrap._x = value;
      container->push_back(value_wrap);
      return *this;
    }
    wrap_insert_iterator& operator* ()
    { return *this; }
    wrap_insert_iterator& operator++ ()
    { return *this; }
    wrap_insert_iterator operator++ (int)
    { return *this; }
  };

  // Vertex construct
  class All_verts_iter {
  public:
    Vertex_iterator _x = Vertex_iterator();
    All_verts_iter() {
      _x = Vertex_iterator();
    }
    All_verts_iter(Vertex_iterator x) { _x = x; }
    All_verts_iter& operator*() { return *this; }
    All_verts_iter& operator++() {
      _x++;
      return *this;
    }
    All_verts_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_verts_iter other) { return (_x == other._x); }
    bool operator!=(All_verts_iter other) { return (_x != other._x); }
    Vertex vertex() { return Vertex((Vertex_handle)(_x)); }
  };
  All_verts_iter all_verts_begin() { return All_verts_iter(T.vertices_begin()); }
  All_verts_iter all_verts_end() { return All_verts_iter(T.vertices_end()); }

  class Vertex {
  public:
    Vertex_handle _x = Vertex_handle();
    Vertex() { _x = Vertex_handle(); }
    Vertex(Vertex_handle x) { _x = x; }
    Vertex(All_verts_iter x) { _x = static_cast<Vertex_handle>(x._x); }
    bool operator==(Vertex other) { return (_x == other._x); }
    bool operator!=(Vertex other) { return (_x != other._x); }
    void point(double* out) {
      Point p = _x->point();
      Cartesian_const_iterator_d it;
      int i = 0;
      for (it = _x->cartesian_begin(); it != _x->cartesian_end(); ++it) {
	out[i] = *it;
	i++;
      }
    }
    Info info() { return _x->data(); }
    Cell cell() { return Cell(_x->full_cell()); }
    void set_cell(Cell c) { _x->set_full_cell(c._x); }
    void set_point(double *x) {
      Point p = pos2point(x);
      _x->set_point(p);
    }
  };


  // Facet construct
  class All_facets_iter {
  public:
    Facet_iterator _x = Facet_iterator();
    All_facets_iter() { _x = Facet_iterator(); }
    All_facets_iter(Facet_iterator x) { _x = x; }
    All_facets_iter& operator*() { return *this; }
    All_facets_iter& operator++() {
      _x++;
      return *this;
    }
    All_facets_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_facets_iter other) { return (_x == other._x); }
    bool operator!=(All_facets_iter other) { return (_x != other._x); }
  };
  All_facets_iter all_facets_begin() { return All_facets_iter(T.facets_begin()); }
  All_facets_iter all_facets_end() { return All_facets_iter(T.facets_end()); }

  class Facet {
  public:
    Facet_handle _x = Facet_handle();
    Facet() {}
    Facet(Facet_handle x) { _x = x; }
    Facet(Facet_iterator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(Finite_facet_iterator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(All_facets_iter x) { _x = static_cast<Facet_handle>(*(x._x)); }
    Facet(Cell x, int i1) { _x = Facet_handle(x._x, i1); }
    Cell cell() const { return Cell(_x.full_cell()); }
    int ind() const { return _x.index_of_covertex(); }
    Vertex vertex(int i) const {
      return Vertex(cell().vertex(ind() + 1 + i));
    }
    bool operator==(Facet other) const { return (_x == other._x); }
    bool operator!=(Facet other) const { return (_x != other._x); }
  };

  // Cell construct
  class All_cells_iter {
  public:
    Cell_iterator _x = Cell_iterator();
    All_cells_iter() {
      _x = Cell_iterator();
    }
    All_cells_iter(Cell_iterator x) { _x = x; }
    All_cells_iter& operator*() { return *this; }
    All_cells_iter& operator++() {
      _x++;
      return *this;
    }
    All_cells_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_cells_iter other) { return (_x == other._x); }
    bool operator!=(All_cells_iter other) { return (_x != other._x); }
  };
  All_cells_iter all_cells_begin() { return All_cells_iter(T.full_cells_begin()); }
  All_cells_iter all_cells_end() { return All_cells_iter(T.full_cells_end()); }

  class Cell {
  public:
    Cell_handle _x = Cell_handle();
    Cell() { _x = Cell_handle(); }
    Cell(Cell_handle x) { _x = x; }
    Cell(All_cells_iter x) { _x = static_cast<Cell_handle>(x._x); }
    bool operator==(Cell other) const { return (_x == other._x); }
    bool operator!=(Cell other) const { return (_x != other._x); }
  
    Vertex vertex(int i) const { return Vertex(_x->vertex(i)); }
    bool has_vertex(Vertex v) const { return _x->has_vertex(v._x); }
    bool has_vertex(Vertex v, int *i) const { return _x->has_vertex(v._x, *i); }
    int ind(Vertex v) const { return _x->index(v._x); }

    Cell neighbor(int i) const { return Cell(_x->neighbor(i)); }
    bool has_neighbor(Cell c) const { return _x->has_neighbor(c._x); }
    bool has_neighbor(Cell c, int *i) const { return _x->has_neighbor(c._x, *i); }
    int ind(Cell c) const { return _x->index(c._x); }

    void set_vertex(int i, Vertex v) { _x->set_vertex(i, v._x); }
    void set_vertices() { _x->set_vertices(); }
    void set_neighbor(int i, Cell c) { _x->set_neighbor(i%4, c._x); }
  };

  

};


