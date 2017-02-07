// TODO:
// - Add support for argbitrary return objects so that dual can be added
// - Line dual_support(Cell c, int i)
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
#if (CGAL_VERSION_NR >= 1040401000)
#include <CGAL/Delaunay_triangulation_cell_base_with_circumcenter_3.h>
#else
#include <CGAL/Triangulation_cell_base_with_circumcenter_3.h>
#endif
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Unique_hash_map.h>
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel            K3;
#if (CGAL_VERSION_NR >= 1040401000)
typedef CGAL::Delaunay_triangulation_cell_base_with_circumcenter_3<K3> Cb3;
#else
typedef CGAL::Triangulation_cell_base_with_circumcenter_3<K3>          Cb3;
#endif

template <typename Info_>
class Delaunay_with_info_3
{
 public:
  typedef CGAL::Delaunay_triangulation_3<K3, CGAL::Triangulation_data_structure_3<CGAL::Triangulation_vertex_base_with_info_3<Info_, K3>, Cb3>> Delaunay;
  typedef typename Delaunay::Point                     Point;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Edge                      Edge_handle;  // not really a handle, just for disambiguation
  typedef typename Delaunay::Facet                     Facet_handle; // not really a handle, just for disambiguation
  typedef typename Delaunay::Cell_handle               Cell_handle;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Edge_iterator             Edge_iterator;
  typedef typename Delaunay::Facet_iterator            Facet_iterator;
  typedef typename Delaunay::Cell_iterator             Cell_iterator;
  typedef typename Delaunay::All_vertices_iterator     All_vertices_iterator;
  typedef typename Delaunay::All_edges_iterator        All_edges_iterator;
  typedef typename Delaunay::All_facets_iterator       All_facets_iterator;
  typedef typename Delaunay::All_cells_iterator        All_cells_iterator;
  typedef typename Delaunay::Finite_vertices_iterator  Finite_vertices_iterator;
  typedef typename Delaunay::Finite_edges_iterator     Finite_edges_iterator;
  typedef typename Delaunay::Finite_facets_iterator    Finite_facets_iterator;
  typedef typename Delaunay::Finite_cells_iterator     Finite_cells_iterator;
  typedef typename Delaunay::Facet_circulator          Facet_circulator;
  typedef typename Delaunay::Cell_circulator           Cell_circulator;
  typedef typename Delaunay::Tetrahedron               Tetrahedron;
  typedef typename Delaunay::Locate_type               Locate_type;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Cell_handle,int>    Cell_hash;
  typedef Info_ Info;
  Delaunay T;
  bool updated = false;
  Delaunay_with_info_3() {};
  Delaunay_with_info_3(double *pts, Info *val, uint32_t n) { insert(pts, val, n); }
  bool is_valid() const { return T.is_valid(); }
  uint32_t num_finite_verts() const { return static_cast<uint32_t>(T.number_of_vertices()); }
  uint32_t num_finite_edges() const { return static_cast<uint32_t>(T.number_of_finite_edges()); }
  uint32_t num_finite_facets() const { return static_cast<uint32_t>(T.number_of_finite_facets()); }
  uint32_t num_finite_cells() const { return static_cast<uint32_t>(T.number_of_finite_cells()); }
  uint32_t num_infinite_verts() const { return 1; }
  uint32_t num_infinite_edges() const { return static_cast<uint32_t>(T.number_of_edges() - T.number_of_finite_edges()); }
  uint32_t num_infinite_facets() const { return static_cast<uint32_t>(T.number_of_facets() - T.number_of_finite_facets()); }
  uint32_t num_infinite_cells() const { return static_cast<uint32_t>(T.number_of_cells() - T.number_of_finite_cells()); }
  uint32_t num_verts() const { return static_cast<uint32_t>(T.number_of_vertices() + num_infinite_verts()); }
  uint32_t num_edges() const { return static_cast<uint32_t>(T.number_of_edges()); }
  uint32_t num_facets() const { return static_cast<uint32_t>(T.number_of_facets()); }
  uint32_t num_cells() const { return static_cast<uint32_t>(T.number_of_cells()); }

  bool is_equal(const Delaunay_with_info_3<Info> other) const {
    // Verts
    if (num_verts() != other.num_verts()) return false;
    if (num_finite_verts() != other.num_finite_verts()) return false;
    if (num_infinite_verts() != other.num_infinite_verts()) return false;
    // Cells
    if (num_cells() != other.num_cells()) return false;
    if (num_finite_cells() != other.num_finite_cells()) return false;
    if (num_infinite_cells() != other.num_infinite_cells()) return false;
    // Edges
    if (num_edges() != other.num_edges()) return false;
    if (num_finite_edges() != other.num_finite_edges()) return false;
    if (num_infinite_edges() != other.num_infinite_edges()) return false;
    // Facets
    if (num_facets() != other.num_facets()) return false;
    if (num_finite_facets() != other.num_finite_facets()) return false;
    if (num_infinite_facets() != other.num_infinite_facets()) return false;
    return true;
  }

  class Vertex;
  class Edge;
  class Facet;
  class Cell;

  Vertex infinite_vertex() const {
    return Vertex(T.infinite_vertex());
  }

  void insert(double *pts, Info *val, uint32_t n)
  {
    updated = true;
    uint32_t i, j;
    std::vector< std::pair<Point,Info> > points;
    for (i = 0; i < n; i++) {
      j = 3*i;
      points.push_back( std::make_pair( Point(pts[j],pts[j+1],pts[j+2]), val[i]) );
    }
    T.insert( points.begin(),points.end() );
  }
  void remove(Vertex v) { updated = true; T.remove(v._x); }
  void clear() { updated = true; T.clear(); }

  Vertex move(Vertex v, double *pos) {
    updated = true;
    Point p = Point(pos[0], pos[1], pos[2]);
    return Vertex(T.move(v._x, p));
  }
  Vertex move_if_no_collision(Vertex v, double *pos) {
    updated = true;
    Point p = Point(pos[0], pos[1], pos[2]);
    return Vertex(T.move_if_no_collision(v._x, p));
  }

  Vertex get_vertex(Info index) const {
    Finite_vertices_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      if (it->info() == index)
        return Vertex(static_cast<Vertex_handle>(it));
    }
    return Vertex(T.infinite_vertex());
  }

  Cell locate(double* pos, int& lt, int& li, int& lj) const {
    Point p = Point(pos[0], pos[1], pos[2]);
    Locate_type lt_out = Locate_type(0);
    Cell out = Cell(T.locate(p, lt_out, li, lj));
    lt = (int)lt_out;
    return out;
  }
  Cell locate(double* pos, int& lt, int& li, int& lj, Cell c) const {
    Point p = Point(pos[0], pos[1], pos[2]);
    Locate_type lt_out = Locate_type(0);
    Cell out = Cell(T.locate(p, lt_out, li, lj, c._x));
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
    All_vertices_iterator _x = All_vertices_iterator();
    All_verts_iter() {
      _x = All_vertices_iterator();
    }
    All_verts_iter(All_vertices_iterator x) { _x = x; }
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
    Vertex vertex() { return Vertex((Vertex_handle)_x); }
  };
  All_verts_iter all_verts_begin() { return All_verts_iter(T.all_vertices_begin()); }
  All_verts_iter all_verts_end() { return All_verts_iter(T.all_vertices_end()); }

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
      out[0] = p.x();
      out[1] = p.y();
      out[2] = p.z();
    }
    Info info() { return _x->info(); }
    Cell cell() { return Cell(_x->cell()); }
    void set_cell(Cell c) { _x->set_cell(c._x); }
    void set_point(double *x) {
      Point p = Point(x[0], x[1], x[2]);
      _x->set_point(p);
    }
  };

  // Edge construct
  class All_edges_iter {
  public:
    All_edges_iterator _x = All_edges_iterator();
    All_edges_iter() { _x = All_edges_iterator(); }
    All_edges_iter(All_edges_iterator x) { _x = x; }
    All_edges_iter& operator*() { return *this; }
    All_edges_iter& operator++() {
      _x++;
      return *this;
    }
    All_edges_iter& operator--() {
      _x--;
      return *this;
    }
    bool operator==(All_edges_iter other) { return (_x == other._x); }
    bool operator!=(All_edges_iter other) { return (_x != other._x); }
  };
  All_edges_iter all_edges_begin() { return All_edges_iter(T.all_edges_begin()); }
  All_edges_iter all_edges_end() { return All_edges_iter(T.all_edges_end()); }

  class Edge {
  public:
    Edge_handle _x = Edge_handle();
    Edge() {}
    Edge(Edge_handle x) { _x = x; }
    Edge(All_edges_iterator x) { _x = static_cast<Edge_handle>(*x); }
    Edge(Finite_edges_iterator x) { _x = static_cast<Edge_handle>(*x); }
    Edge(All_edges_iter x) { _x = static_cast<Edge_handle>(*(x._x)); }
    Edge(Cell x, int i1, int i2) { _x = Edge_handle(x._x, i1%4, i2%4); }
    Cell cell() const { return Cell(_x.first); }
    int ind1() const { return _x.second; }
    int ind2() const { return _x.third; }
    Vertex vertex(int i) {
      if ((i % 2) == 0)
	return v1();
      else
	return v2();
    }
    Vertex_handle _v1() const { return _x.first->vertex(_x.second); }
    Vertex_handle _v2() const { return _x.first->vertex(_x.third); }
    Vertex v1() const { return Vertex(_v1()); }
    Vertex v2() const { return Vertex(_v2()); }
    bool operator==(Edge other) const { return (_x == other._x); }
    bool operator!=(Edge other) const { return (_x != other._x); }
  };


  // Facet construct
  class All_facets_iter {
  public:
    All_facets_iterator _x = All_facets_iterator();
    All_facets_iter() { _x = All_facets_iterator(); }
    All_facets_iter(All_facets_iterator x) { _x = x; }
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
  All_facets_iter all_facets_begin() { return All_facets_iter(T.all_facets_begin()); }
  All_facets_iter all_facets_end() { return All_facets_iter(T.all_facets_end()); }

  class Facet {
  public:
    Facet_handle _x = Facet_handle();
    Facet() {}
    Facet(Facet_handle x) { _x = x; }
    Facet(All_facets_iterator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(Finite_facets_iterator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(Facet_circulator x) { _x = static_cast<Facet_handle>(*x); }
    Facet(All_facets_iter x) { _x = static_cast<Facet_handle>(*(x._x)); }
    Facet(Cell x, int i1) { _x = Facet_handle(x._x, i1); }
    Cell cell() const { return Cell(_x.first); }
    int ind() const { return _x.second; }
    Vertex vertex(int i) const { 
      return Vertex(cell().vertex(ind() + 1 + (i%3))); 
    }
    Edge edge(int i) const {
      int i1 = ind() + 1 + ((i+1)%3);
      int i2 = ind() + 1 + ((i+2)%3);
      return Edge(cell(), i1, i2);
    }
    bool operator==(Facet other) const { return (_x == other._x); }
    bool operator!=(Facet other) const { return (_x != other._x); }
  };


  // Cell construct
  class All_cells_iter {
  public:
    All_cells_iterator _x = All_cells_iterator();
    All_cells_iter() {
      _x = All_cells_iterator();
    }
    All_cells_iter(All_cells_iterator x) { _x = x; }
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
  All_cells_iter all_cells_begin() { return All_cells_iter(T.all_cells_begin()); }
  All_cells_iter all_cells_end() { return All_cells_iter(T.all_cells_end()); }

  class Cell {
  public:
    Cell_handle _x = Cell_handle();
    Cell() { _x = Cell_handle(); }
    Cell(Cell_handle x) { _x = x; }
    Cell(Cell_circulator x) { _x = static_cast<Cell_handle>(x); }
    Cell(All_cells_iter x) { _x = static_cast<Cell_handle>(x._x); }
    Cell(Vertex v1, Vertex v2, Vertex v3, Vertex v4) {
      _x = Cell_handle(v1._x, v2._x, v3._x, v4._x); }
    Cell(Vertex v1, Vertex v2, Vertex v3, Vertex v4, 
	 Cell n1, Cell n2, Cell n3, Cell n4) {
      _x = Cell_handle(v1._x, v2._x, v3._x, v4._x,
		       n1._x, n2._x, n3._x, n4._x);
    }
    bool operator==(Cell other) const { return (_x == other._x); }
    bool operator!=(Cell other) const { return (_x != other._x); }

    Facet facet(int i) const { return Facet(*this, i); }

    Vertex vertex(int i) const { return Vertex(_x->vertex(i%4)); }
    bool has_vertex(Vertex v) const { return _x->has_vertex(v._x); }
    bool has_vertex(Vertex v, int *i) const { return _x->has_vertex(v._x, *i); }
    int ind(Vertex v) const { return _x->index(v._x); }

    Cell neighbor(int i) const { return Cell(_x->neighbor(i%4)); }
    bool has_neighbor(Cell c) const { return _x->has_neighbor(c._x); }
    bool has_neighbor(Cell c, int *i) const { return _x->has_neighbor(c._x, *i); }
    int ind(Cell c) const { return _x->index(c._x); }

    void set_vertex(int i, Vertex v) { _x->set_vertex(i%4, v._x); }
    void set_vertices() { _x->set_vertices(); }
    void set_vertices(Vertex v1, Vertex v2, Vertex v3, Vertex v4) {
      _x->set_vertices(v1._x, v2._x, v3._x, v4._x); }
    void set_neighbor(int i, Cell c) { _x->set_neighbor(i%4, c._x); }
    void set_neighbors() { _x->set_neighbors(); }
    void set_neighbors(Cell c1, Cell c2, Cell c3, Cell c4) {
      _x->set_neighbors(c1._x, c2._x, c3._x, c4._x); 
    }

    double min_angle() const {
      Point p0, p1, p2, p3;
      CGAL::Vector_3<K3> v1, v2, v3;
      double min_angle = 99999999999999;
      double theta1, theta2, theta3, theta0;
      double tangent, angle;
      for (int i = 0; i < 4; i++) {
	p0 = _x->vertex(i)->point();
	p1 = _x->vertex((i+1)%4)->point();
	p2 = _x->vertex((i+2)%4)->point();
	p3 = _x->vertex((i+3)%4)->point();
	v1 = p1 - p0;
	v2 = p2 - p0;
	v3 = p3 - p0;
	theta1 = std::abs(v2 * v3 / CGAL::sqrt(v2*v2) / CGAL::sqrt(v3*v3));
	theta2 = std::abs(v3 * v1 / CGAL::sqrt(v3*v3) / CGAL::sqrt(v1*v1));
	theta3 = std::abs(v1 * v2 / CGAL::sqrt(v1*v1) / CGAL::sqrt(v2*v2));
	theta0 = (theta1 + theta2 + theta3)/2.0;
	tangent = std::sqrt(std::tan(theta0/2.0)*
			    std::tan((theta0 - theta1)/2.0)*
			    std::tan((theta0 - theta2)/2.0)*
			    std::tan((theta0 - theta3)/2.0));
	angle = 4.0*std::atan(tangent);
	if (angle < min_angle)
	  min_angle = angle;
      }
      return min_angle;
    }
      
  };

  bool are_equal(const Facet f1, const Facet f2) const {
    return T.are_equal(f1._x, f2._x);
    // Vertex x1 = f1.vertex(0), x2 = f1.vertex(1), x3 = f1.vertex(2);
    // Vertex o1 = f2.vertex(0), o2 = f2.vertex(1), o3 = f2.vertex(2);
    // if (x1 == o1) {
    //   if ((x2 == o2) && (x3 == o3))
    // 	return true;
    //   else if ((x2 == o3) && (x3 == o2))
    // 	return true;
    //   else
    // 	return false;
    // } else if (x1 == o2) {
    //   if ((x2 == o1) && (x3 == o3))
    // 	return true;
    //   else if ((x2 == o3) && (x3 == o1))
    // 	return true;
    //   else
    // 	return false;
    // } else if (x1 == o3) {
    //   if ((x2 == o2) && (x3 == o1))
    // 	return true;
    //   else if ((x2 == o1) && (x3 == o2))
    // 	return true;
    //   else
    // 	return false;
    // } else
    //   return false;
  }
  bool are_equal(const Edge e1, const Edge e2) const {
    if ((e1.v1() == e2.v1()) && (e1.v2() == e2.v2()))
      return true;
    else if ((e1.v1() == e2.v2()) && (e1.v2() == e2.v1()))
      return true;
    else
      return false;
  }

  // Testing incidence to the infinite vertex
  bool is_infinite(Vertex x) const { return T.is_infinite(x._x); }
  bool is_infinite(Edge x) const { return T.is_infinite(x._x); }
  bool is_infinite(Facet x) const { return T.is_infinite(x._x); }
  bool is_infinite(Cell x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_verts_iter x) const { return T.is_infinite(x._x); }
  bool is_infinite(All_edges_iter x) const { 
    const Edge_iterator e = x._x;
    return T.is_infinite(*e);
  }
  bool is_infinite(All_facets_iter x) const {
    const Facet_iterator f = x._x;
    return T.is_infinite(*f);
  }
  bool is_infinite(All_cells_iter x) const { return T.is_infinite(x._x); }

  bool is_edge(Vertex x1, Vertex x2, Cell& c, int& i, int& j) const { 
    return T.is_edge(x1._x, x2._x, c._x, i, j); 
  }
  bool is_facet(Vertex x1, Vertex x2, Vertex x3, Cell& c, int& i, int& j, int& k) const {
    return T.is_facet(x1._x, x2._x, x3._x, c._x, i, j, k);
  }
  bool is_cell(Vertex x1, Vertex x2, Vertex x3, Vertex x4, 
	       Cell& c, int& i1, int& i2, int& i3, int& i4) const { 
    return T.is_cell(x1._x, x2._x, x3._x, x4._x, c._x, i1, i2, i3, i4);
  }

  // Parts incident to a vertex
  std::vector<Vertex> incident_vertices(Vertex x) const {
    std::vector<Vertex> out;
    T.adjacent_vertices(x._x, wrap_insert_iterator<Vertex,Vertex_handle>(out));
    return out;
  }
  std::vector<Edge> incident_edges(Vertex x) const {
    std::vector<Edge> out;
    T.incident_edges(x._x, wrap_insert_iterator<Edge,Edge_handle>(out));
    return out;
  }
  std::vector<Facet> incident_facets(Vertex x) const {
    std::vector<Facet> out;
    T.incident_facets(x._x, wrap_insert_iterator<Facet,Facet_handle>(out));
    return out;
  }
  std::vector<Cell> incident_cells(Vertex x) const {
    std::vector<Cell> out;
    T.incident_cells(x._x, wrap_insert_iterator<Cell,Cell_handle>(out));
    return out;
  }

  // Parts incident to an edge
  std::vector<Vertex> incident_vertices(Edge x) const {
    std::vector<Vertex> out;
    out.push_back(x.v1());
    out.push_back(x.v2());
    return out;
  }
  std::vector<Edge> incident_edges(Edge x) const {
    uint32_t i;
    std::vector<Edge> out1, out2, out;
    T.incident_edges(x.v1()._x, wrap_insert_iterator<Edge,Edge_handle>(out1));
    T.incident_edges(x.v2()._x, wrap_insert_iterator<Edge,Edge_handle>(out2));
    for (i = 0; i < out1.size(); i++) {
      if (!(are_equal(x, out1[i])))
	out.push_back(out1[i]);
    }
    for (i = 0; i < out2.size(); i++) {
      if (!(are_equal(x, out2[i])))
	out.push_back(out2[i]);
    }
    return out;
  }
  std::vector<Facet> incident_facets(Edge x) const {
    std::vector<Facet> out;
    Facet_circulator cc = T.incident_facets(x._x), done(cc);
    if (cc == 0)
      return out;
    do {
      out.push_back(Facet(cc));
    } while (++cc != done);
    return out;
  }
  std::vector<Cell> incident_cells(Edge x) const {
    std::vector<Cell> out;
    Cell_circulator cc = T.incident_cells(x._x), done(cc);
    if (cc == 0)
      return out;
    do {
      out.push_back(Cell(cc));
    } while (++cc != done);
    return out;
  }

  // Constructs incident to a facet
  std::vector<Vertex> incident_vertices(Facet x) const {
    std::vector<Vertex> out;
    for (int i = 0; i < 3; i++) {
      out.push_back(x.vertex(i));
    }
    return out;
  }
  std::vector<Edge> incident_edges(Facet x) const {
    std::vector<Edge> out;
    for (int i = 0; i < 3; i++) {
      out.push_back(x.edge(i));
    }
    return out;
  }
  std::vector<Facet> incident_facets(Facet x) const {
    std::vector<Facet> out;
    std::vector<Edge> edges = incident_edges(x);
    for (uint32_t i = 0; i < edges.size(); i++) {
      Facet_circulator cc = T.incident_facets(edges[i]._x), done(cc);
      if (cc != 0) {
	do {
	  if (!(are_equal(Facet(cc), x)))
	    out.push_back(Facet(cc));
	} while (++cc != done);
      }
    } 
    return out;
  }
  std::vector<Cell> incident_cells(Facet x) const {
    std::vector<Cell> out;
    out.push_back(x.cell());
    out.push_back(x.cell().neighbor(x.ind()));
    return out;
  }

  // Constructs incident to a cell
  std::vector<Vertex> incident_vertices(Cell x) const {
    std::vector<Vertex> out;
    for (int i = 0; i < 4; i++)
      out.push_back(x.vertex(i));
    return out;
  }
  std::vector<Edge> incident_edges(Cell x) const {
    std::vector<Edge> out;
    int i1, i2;
    for (i1 = 0; i1 < 4; i1++) {
      for (i2 = (i1+1); i2 < 4; i2++) {
	out.push_back(Edge(x, i1, i2));
      }
    }
    return out;
  }
  std::vector<Facet> incident_facets(Cell x) const {
    std::vector<Facet> out;
    for (int i = 0; i < 4; i++)
      out.push_back(Facet(x, i));
    return out;
  }
  std::vector<Cell> incident_cells(Cell x) const {
    std::vector<Cell> out;
    for (int i = 0; i < 4; i++)
      out.push_back(x.neighbor(i));
    return out;
  }

  Vertex nearest_vertex(double* pos) const {
    Point p = Point(pos[0], pos[1], pos[2]);
    Vertex out = Vertex(T.nearest_vertex(p));
    return out;
  }

  Facet mirror_facet(Facet x) const { return Facet(T.mirror_facet(x._x)); }
  int mirror_index(Cell x, int i) const { return T.mirror_index(x._x, i); }
  Vertex mirror_vertex(Cell x, int i) const { return Vertex(T.mirror_vertex(x._x, i)); }

  void circumcenter(Cell x, double* out) const {
    if (T.is_infinite(x._x)) {
      out[0] = std::numeric_limits<double>::infinity();
      out[1] = std::numeric_limits<double>::infinity();
      out[2] = std::numeric_limits<double>::infinity();
    } else {
      Point p = x._x->circumcenter();
      out[0] = p.x();
      out[1] = p.y();
      out[2] = p.z();
    }
  }

  double dual_volume(const Vertex v) const {
    std::list<Edge_handle> edges;
    T.incident_edges(v._x, std::back_inserter(edges));

    Point orig = v._x->point();
    double vol = 0.0;
    for (typename std::list<Edge_handle>::iterator eit = edges.begin() ;
         eit != edges.end() ; ++eit) {

      Facet_circulator fstart = T.incident_facets(*eit);
      Facet_circulator fcit = fstart;
      std::vector<Point> pts;
      do {
	if (T.is_infinite(fcit->first))
	  return -1.0;
        Point dual_orig = fcit->first->circumcenter();
        pts.push_back(dual_orig);
        ++fcit;
      } while (fcit != fstart);

      for (uint32_t i=1 ; i<pts.size()-1 ; i++)
        vol += Tetrahedron(orig,pts[0],pts[i],pts[i+1]).volume();
    }
    return vol;
  }

  void dual_volumes(double *vols) const {
    for (Finite_vertices_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      vols[it->info()] = dual_volume(Vertex(it));
    }    
  }

  bool is_boundary_cell(const Cell c) const {
    if (T.is_infinite(c._x))
      return true;
    for (int i = 0; i < 4; i++) {
      if (T.is_infinite(c.neighbor(i)._x))
        return true;
    }
    return false;
  }
  
  double minimum_angle(const Cell c) const {
    return c.min_angle();
  }
  
  int minimum_angles(double* angles) const {
    int i = 0;
    for (All_cells_iterator it = T.all_cells_begin(); it != T.all_cells_end(); it++) {
      if (!(is_boundary_cell(Cell(it)))) {
        angles[i] = minimum_angle(Cell(it));
        i++;
      }
    }
    return i;
  }

  double length(const Edge e) const {
    if (is_infinite(e))
      return -1.0;
    Vertex_handle v1 = e.v1()._x;
    Vertex_handle v2 = e.v2()._x;
    Point p1 = v1->point();
    Point p2 = v2->point();
    double out = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, p2)));
    return out;
  }

  bool flip(Cell x, int i, int j) { updated = true; return T.flip(x._x, i, j); }
  bool flip(Edge x) { updated = true; return T.flip(x.cell()._x, x.ind1(), x.ind2()); }
  bool flip(Cell x, int i) { updated = true; return T.flip(x._x, i); }
  bool flip(Facet x) { updated = true; return T.flip(x.cell()._x, x.ind()); }
  void flip_flippable(Cell x, int i, int j) { updated = true; T.flip_flippable(x._x, i, j); }
  void flip_flippable(Edge x) { updated = true; T.flip_flippable(x.cell()._x, x.ind1(), x.ind2()); }
  void flip_flippable(Cell x, int i) { updated = true; T.flip_flippable(x._x, i); }
  void flip_flippable(Facet x) { updated = true; T.flip_flippable(x.cell()._x, x.ind()); }

  std::pair<std::vector<Cell>,std::vector<Facet>>
    find_conflicts(double* pos, Cell start) const {
    std::pair<std::vector<Cell>,std::vector<Facet>> out;
    std::vector<Cell> cit;
    std::vector<Facet> fit;
    Point p = Point(pos[0], pos[1], pos[2]);
    T.find_conflicts(p, start._x, 
		     wrap_insert_iterator<Facet,Facet_handle>(fit),
		     wrap_insert_iterator<Cell,Cell_handle>(cit));
    out = std::make_pair(cit, fit);
    return out;
  }

  int side_of_cell(const double* pos, Cell c, int& lt, int& li, int& lj) const {
    if (T.is_infinite(c._x))
      return -1;
    else if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
      return 1;
    else {
      Point p = Point(pos[0], pos[1], pos[2]);
      Locate_type lt_out = Locate_type(0);    
      int out = -(int)T.side_of_cell(p, c._x, lt_out, li, lj);
      lt = (int)lt_out;
      return out;
    }
  }
  int side_of_edge(const double* pos, const Edge e, int& lt, int& li) const {
    if (T.is_infinite(e._x))
      return -1;
    else if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
      return 1;
    else {
      Point p = Point(pos[0], pos[1], pos[2]);
      Locate_type lt_out = Locate_type(0);
      int out = -(int)T.side_of_edge(p, e._x, lt_out, li);
      lt = (int)lt_out;
      return out;
    }
  }
  int side_of_facet(const double* pos, const Facet f, int& lt, int& li, int& lj) const {
    if (T.is_infinite(f._x))
      return -1;
    else if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
      return 1;
    else {
      Point p = Point(pos[0], pos[1], pos[2]);
      Locate_type lt_out = Locate_type(0);
      int out = -(int)T.side_of_facet(p, f._x, lt_out, li, lj);
      lt = (int)lt_out;
      return out;
    }
  }

  // Currently segfaults inside CGAL function call
  // int side_of_circle(const Facet f, const double* pos) const {
  //   if (T.is_infinite(f._x))
  //     return -1;
  //   else if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
  //     return 1;
  //   else {
  //     Point p = Point(pos[0], pos[1], pos[2]);
  //     return -(int)(T.side_of_circle(f.cell()._x, f.ind(), p));
  //     // return (int)(-T.side_of_circle(f._x, p));
  //   }
  // }
  int side_of_sphere(const Cell c, const double* pos) const {
    if (T.is_infinite(c._x))
      return -1;
    else if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
      return 1;
    else {
      Point p = Point(pos[0], pos[1], pos[2]);
      return -(int)T.side_of_sphere(c._x, p);
    }
  }

  bool is_Gabriel(const Edge e) { return T.is_Gabriel(e._x); }
  bool is_Gabriel(const Facet f) { return T.is_Gabriel(f._x); }

  void write_to_file(const char* filename) const
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      write_to_buffer(os);
      os.close();
    }
  }

  void write_to_buffer(std::ofstream &os) const {
    // Header
    int n = static_cast<int>(T.number_of_vertices());
    int m = static_cast<int>(T.tds().number_of_cells());
    int d = static_cast<int>(T.dimension());
    os.write((char*)&n, sizeof(int));
    os.write((char*)&m, sizeof(int));
    os.write((char*)&d, sizeof(int));
    if (n==0) {
      return;
    }
    // printf("Wrote %d vertices, %d cells, for %d dimensions\n",n,m,d);
    
    Vertex_hash V;
    Cell_hash C;
    
    // first (infinite) vertex 
    int inum = 0;
    Vertex_handle v = T.infinite_vertex();
    if ( v != Vertex_handle()) {
      V[v] = inum++;
    }
    
    // other vertices
    Info info;
    double x, y, z;
    for( All_vertices_iterator vit = T.tds().vertices_begin(); vit != T.tds().vertices_end() ; ++vit) {
      if ( v != vit ) {
	V[vit] = inum++;
	info = static_cast<Info>(vit->info());
	x = static_cast<double>(vit->point().x());
	y = static_cast<double>(vit->point().y());
	z = static_cast<double>(vit->point().z());
	os.write((char*)&info, sizeof(Info));
	os.write((char*)&x, sizeof(double));
	os.write((char*)&y, sizeof(double));
	os.write((char*)&z, sizeof(double));
      }
    }
    // printf("%d nverts, %d inum\n",T.number_of_vertices(),inum);
    
    // vertices of the cells
    inum = 0;
    int dim = (d == -1 ? 1 :  d + 1);
    int index;
    for( Cell_iterator ib = T.tds().cells_begin(); 
	 ib != T.tds().cells_end(); ++ib) {
      C[ib] = inum++;
      for(int j = 0; j < dim ; ++j) {
	index = V[ib->vertex(j)];
	os.write((char*)&index, sizeof(int));
      }
    }
    // printf("%d ncells, %d inum\n",T.tds().number_of_cells(),inum);
    
    // neighbor pointers of the cells
    for( Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      for(int j = 0; j < d+1; ++j){
	index = C[it->neighbor(j)];
	os.write((char*)&index, sizeof(int));
      }
    }

  }


  void read_from_file(const char* filename)
  {
    std::ifstream is(filename, std::ios::binary);
    if (!is) std::cerr << "Error cannot open file: " << filename << std::endl;
    else {
      read_from_buffer(is);
      is.close();
    }
  }

  void read_from_buffer(std::ifstream &is) {
    
    updated = true;
    if (T.number_of_vertices() != 0)  
      T.clear();
    
    // header
    int n, m, d;
    is.read((char*)&n, sizeof(int));
    is.read((char*)&m, sizeof(int));
    is.read((char*)&d, sizeof(int));
    
    if (n==0) {
      return;
    }
    
    T.tds().set_dimension(d);
    All_cells_iterator to_delete = T.tds().cells_begin();
    
    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);
    
    // infinite vertex
    int i = 0;
    V[0] = T.infinite_vertex();
    ++i;

    // read vertices
    Info info;
    double x, y, z;
    for( ; i <= n; ++i) {
      V[i] = T.tds().create_vertex();
      is.read((char*)&info, sizeof(Info));
      is.read((char*)&x, sizeof(double));
      is.read((char*)&y, sizeof(double));
      is.read((char*)&z, sizeof(double));
      (*(V[i])).point() = Point(x,y,z);
      (*(V[i])).info() = info;
    }
    
    // Creation of the cells
    int index;
    int dim = (d == -1 ? 1 : d + 1);
    {
      for(i = 0; i < m; ++i) {
	C[i] = T.tds().create_cell() ;
	for(int j = 0; j < dim ; ++j){
	  is.read((char*)&index, sizeof(int));
	  C[i]->set_vertex(j, V[index]);
	  V[index]->set_cell(C[i]);
	}
      }
    }
    
    // Setting the neighbor pointers
    {
      for(i = 0; i < m; ++i) {
	for(int j = 0; j < d+1; ++j){
	  is.read((char*)&index, sizeof(int));
	  C[i]->set_neighbor(j, C[index]);
	}
      }
    }
    
    // delete flat cell
    T.tds().delete_cell(to_delete);
    
  }

  template <typename I>
  I serialize(I &n, I &m, int32_t &d,
              double* vert_pos, Info* vert_info,
              I* cells, I* neighbors) const
  {
    I idx_inf = std::numeric_limits<I>::max();

    // Header
    n = static_cast<int>(T.number_of_vertices());
    m = static_cast<int>(T.tds().number_of_cells());
    d = static_cast<int>(T.dimension());
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Vertex_hash V;
    Cell_hash C;
      
    // first (infinite) vertex 
    Vertex_handle vit;
    int inum = 0;
    Vertex_handle v = T.infinite_vertex();
    V[v] = -1;
    
    // other vertices
    for( All_vertices_iterator vit = T.tds().vertices_begin(); vit != T.tds().vertices_end() ; ++vit) {
      if ( v != vit ) {
	vert_pos[d*inum + 0] = static_cast<double>(vit->point().x());
	vert_pos[d*inum + 1] = static_cast<double>(vit->point().y());
	vert_pos[d*inum + 2] = static_cast<double>(vit->point().z());
	vert_info[inum] = vit->info();
	V[vit] = inum++;
      }
    }
    
    // vertices of the cells
    inum = 0;
    for( Cell_iterator ib = T.tds().cells_begin(); 
	 ib != T.tds().cells_end(); ++ib) {
      for (int j = 0; j < dim ; ++j) {
	vit = ib->vertex(j);
	if ( v == vit )
	  cells[dim*inum + j] = idx_inf;
	else
	  cells[dim*inum + j] = V[vit];
      }
      C[ib] = inum++;
    }
  
    // neighbor pointers of the cells
    inum = 0;
    for( Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      for (int j = 0; j < d+1; ++j){
	neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
      }
      inum++;
    }
    return idx_inf;
  }

  template <typename I>
  Info serialize_idxinfo(I &n, I &m, int32_t &d,
			 Info* cells, I* neighbors) const
  {
    Info idx_inf = std::numeric_limits<Info>::max();

    // Header
    n = static_cast<int>(T.number_of_vertices());
    m = static_cast<int>(T.tds().number_of_cells());
    d = static_cast<int>(T.dimension());
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Cell_hash C;
      
    // first (infinite) vertex 
    Vertex_handle vit;
    int inum = 0;
    Vertex_handle v = T.infinite_vertex();
    
    // vertices of the cells
    inum = 0;
    for( Cell_iterator ib = T.tds().cells_begin(); 
	 ib != T.tds().cells_end(); ++ib) {
      for (int j = 0; j < dim ; ++j) {
	vit = ib->vertex(j);
	if ( v == vit )
	  cells[dim*inum + j] = idx_inf;
	else
	  cells[dim*inum + j] = vit->info();
      }
      C[ib] = inum++;
    }
  
    // neighbor pointers of the cells
    inum = 0;
    for( Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      for (int j = 0; j < d+1; ++j){
	neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
      }
      inum++;
    }
    return idx_inf;
  }

  template <typename I>
  I serialize_info2idx(I &n, I &m, int32_t &d,
		       I* cells, I* neighbors,
		       Info max_info, I* idx) const
  {
    I idx_inf = std::numeric_limits<I>::max();

    // Header
    n = static_cast<I>(max_info);
    d = static_cast<int>(T.dimension());
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Cell_hash C;
      
    // first (infinite) vertex 
    Vertex_handle vit;
    Vertex_handle v = T.infinite_vertex();
    
    // vertices of the cells
    int j;
    bool *include_cell = (bool*)malloc(m*sizeof(bool));
    I inum = 0, inum_tot = 0;
    for( Cell_iterator ib = T.tds().cells_begin(); 
	 ib != T.tds().cells_end(); ++ib) {
      include_cell[inum_tot] = false;
      for (j = 0; j < dim ; ++j) {
        vit = ib->vertex(j);
	// if ((v != vit) and (vit->info() < max_info)) {
	//   include_cell[inum_tot] = true;
	//   break;
	// }
        if ( v == vit) {
          include_cell[inum_tot] = false;
          break;
        } else if (vit->info() < max_info) {
          include_cell[inum_tot] = true;
        }
      }
      if (include_cell[inum_tot]) {
	for (j = 0; j < dim ; ++j) {
	  vit = ib->vertex(j);
	  if ( v == vit )
	    cells[dim*inum + j] = idx_inf;
	  else
	    cells[dim*inum + j] = idx[vit->info()];
	}
	C[ib] = inum++;
      } else {
        C[ib] = idx_inf;
      }
      inum_tot++;
    }
    m = inum;
  
    // neighbor pointers of the cells
    inum = 0, inum_tot = 0;
    for( Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      if (include_cell[inum_tot]) {
	for (int j = 0; j < d+1; ++j){
	  neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
	}
	inum++;
      }
      inum_tot++;
    }

    free(include_cell);
    return idx_inf;
  }

  template <typename I>
  void deserialize(I n, I m, int32_t d,
                   double* vert_pos, Info* vert_info,
                   I* cells, I* neighbors, I idx_inf)
  {
    updated = true;

    if (T.number_of_vertices() != 0)  
      T.clear();
 
    if (n==0) {
      return;
    }

    T.tds().set_dimension(d);

    All_cells_iterator to_delete = T.tds().cells_begin();

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);

    // infinite vertex
    V[n] = T.infinite_vertex();

    // read vertices
    I i;
    for(i = 0; i < n; ++i) {
      V[i] = T.tds().create_vertex();
      V[i]->point() = Point(vert_pos[d*i], vert_pos[d*i + 1], vert_pos[d*i + 2]);
      V[i]->info() = vert_info[i];
    }

    // Creation of the cells
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    for(i = 0; i < m; ++i) {
      C[i] = T.tds().create_cell() ;
      for(int j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
        if (index == idx_inf)
          v = V[n];
        else
          v = V[index];
        C[i]->set_vertex(j, v);
        v->set_cell(C[i]);
      }
    }

    // Setting the neighbor pointers
    for(i = 0; i < m; ++i) {
      for(int j = 0; j < d+1; ++j){
        index = neighbors[(d+1)*i + j];
        C[i]->set_neighbor(j, C[index]);
      }
    }

    // delete flat cell
    T.tds().delete_cell(to_delete);

  }

  template <typename I>
  void deserialize_idxinfo(I n, I m, int32_t d, double* vert_pos, 
			   I* cells, I* neighbors, I idx_inf)
  {
    updated = true;

    if (T.number_of_vertices() != 0)  
      T.clear();
 
    if (n==0) {
      return;
    }

    T.tds().set_dimension(d);

    All_cells_iterator to_delete = T.tds().cells_begin();

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);

    // infinite vertex
    V[n] = T.infinite_vertex();

    // read vertices
    I i;
    for(i = 0; i < n; ++i) {
      V[i] = T.tds().create_vertex();
      V[i]->point() = Point(vert_pos[d*i], vert_pos[d*i + 1], vert_pos[d*i + 2]);
      V[i]->info() = (Info)(i);
    }

    // Creation of the cells
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    for(i = 0; i < m; ++i) {
      C[i] = T.tds().create_cell() ;
      for(int j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
        if (index == idx_inf)
          v = V[n];
        else
          v = V[index];
        C[i]->set_vertex(j, v);
        v->set_cell(C[i]);
      }
    }

    // Setting the neighbor pointers
    for(i = 0; i < m; ++i) {
      for(int j = 0; j < d+1; ++j){
        index = neighbors[(d+1)*i + j];
        C[i]->set_neighbor(j, C[index]);
      }
    }

    // delete flat cell
    T.tds().delete_cell(to_delete);

  }

  void info_ordered_vertices(double* pos) const {
    Info i;
    Point p;
    for (Finite_vertices_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      i = it->info();
      p = it->point();
      pos[3*i + 0] = p.x();
      pos[3*i + 1] = p.y();
      pos[3*i + 2] = p.z();
    }
  }

  void vertex_info(Info* verts) const {
    int i = 0;
    for (Finite_vertices_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      verts[i] = it->info();
      i++;
    }
  }
  
  void edge_info(Info* edges) const {
    int i = 0;
    Info i1, i2;
    for (Finite_edges_iterator it = T.finite_edges_begin(); it != T.finite_edges_end(); it++) {
      i1 = it->first->vertex(it->second)->info();
      i2 = it->first->vertex(it->third)->info();
      edges[2*i + 0] = i1;
      edges[2*i + 1] = i2;
      i++;
    }
  }

  bool intersect_sph_box(Point *c, double r, double *le, double *re) const {
    // x
    if (c->x() < le[0]) {
      if ((c->x() + r) < le[0])
        return false;
    } else if (c->x() > re[0]) {
      if ((c->x() - r) > re[0])
        return false;
    }
    // y
    if (c->y() < le[1]) {
      if ((c->y() + r) < le[1])
        return false;
    } else if (c->y() > re[1]) {
      if ((c->y() - r) > re[1])
        return false;
    }
    // z
    if (c->z() < le[2]) {
      if ((c->z() + r) < le[2])
        return false;
    } else if (c->z() > re[2]) {
      if ((c->z() - r) > re[2])
        return false;
    }
    return true;
  }

  std::vector<std::vector<Info>> outgoing_points(uint64_t nbox,
                                                 double *left_edges,
                                                 double *right_edges) const {
    std::vector<std::vector<Info>> out;
    uint64_t b;
    for (b = 0; b < nbox; b++)
      out.push_back(std::vector<Info>());

    Vertex_handle v;
    Point cc, p1;
    double cr;
    int i, iinf = 0;

    for (All_cells_iterator it = T.all_cells_begin(); it != T.all_cells_end(); it++) {
      if (T.is_infinite(it) == true) {
        // Find index of infinite vertex
        for (i = 0; i < 4; i++) {
          v = it->vertex(i);
          if (T.is_infinite(v)) {
            iinf = i;
            break;
          }
        }
        for (b = 0; b < nbox; b++)
          for (i = 1; i < 4; i++) out[b].push_back((it->vertex((iinf+i)%4))->info());
      } else {
        p1 = it->vertex(0)->point();
	cc = it->circumcenter();
        cr = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, cc)));
        for (b = 0; b < nbox; b++) {
          if (intersect_sph_box(&cc, cr, left_edges + 3*b, right_edges + 3*b))
            for (i = 0; i < 4; i++) out[b].push_back((it->vertex(i))->info());
        }
      }
    }
    for (b = 0; b < nbox; b++) {
      std::sort( out[b].begin(), out[b].end() );
      out[b].erase( std::unique( out[b].begin(), out[b].end() ), out[b].end() );
    }

    return out;
  }

  void boundary_points(double *left_edge, double *right_edge, bool periodic,
                       std::vector<Info>& lx, std::vector<Info>& ly, std::vector<Info>& lz,
                       std::vector<Info>& rx, std::vector<Info>& ry, std::vector<Info>& rz,
                       std::vector<Info>& alln)
  {
    Vertex_handle v;
    Info hv;
    Point cc, p1, p2;
    double cr, icr;
    int i, iinf;

    for (All_cells_iterator it = T.all_cells_begin(); it != T.all_cells_end(); it++) {
      if (T.is_infinite(it) == true) {
        for (i = 0; i < 4; i++) {
          v = it->vertex(i);
          if (T.is_infinite(v)) {
            iinf = i;
            break;
          }
        }
        cc = it->vertex((iinf+1) % 4)->point();
        p1 = it->vertex((iinf+2) % 4)->point();
        p2 = it->vertex((iinf+3) % 4)->point();
	cr = 0;
        icr = std::sqrt(static_cast<double>(CGAL::squared_distance(cc, p1)));
	if (icr > cr) cr = icr;
        icr = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, p2)));
	if (icr > cr) cr = icr;
        icr = std::sqrt(static_cast<double>(CGAL::squared_distance(p2, cc)));
	if (icr > cr) cr = icr;
	cc = Point((cc.x()+p1.x()+p2.x())/3.0, 
		   (cc.y()+p1.y()+p2.y())/3.0, 
		   (cc.z()+p1.z()+p2.z())/3.0);
        if ((cc.x() + cr) > right_edge[0])
          for (i = 1; i < 4; i++) rx.push_back((it->vertex((iinf+i)%4))->info());
        if ((cc.y() + cr) > right_edge[1])
          for (i = 1; i < 4; i++) ry.push_back((it->vertex((iinf+i)%4))->info());
        if ((cc.z() + cr) > right_edge[2])
          for (i = 1; i < 4; i++) rz.push_back((it->vertex((iinf+i)%4))->info());
        if ((cc.x() - cr) < left_edge[0])
          for (i = 1; i < 4; i++) lx.push_back((it->vertex((iinf+i)%4))->info());
        if ((cc.y() - cr) < left_edge[1])
          for (i = 1; i < 4; i++) ly.push_back((it->vertex((iinf+i)%4))->info());
        if ((cc.z() - cr) < left_edge[2])
          for (i = 1; i < 4; i++) lz.push_back((it->vertex((iinf+i)%4))->info());

        // if (periodic == false) {
        //   for (i = 0; i < 4; i++) {
        //     v = it->vertex(i);
        //     if (T.is_infinite(v) == false) {
        //       hv = v->info();
        //       alln.push_back(hv);
        //     }
        //   }
        // }
      } else {
        p1 = it->vertex(0)->point();
	cc = it->circumcenter();
        cr = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, cc)));
        if ((cc.x() + cr) > right_edge[0])
          for (i = 0; i < 4; i++) rx.push_back((it->vertex(i))->info());
        if ((cc.y() + cr) > right_edge[1])
          for (i = 0; i < 4; i++) ry.push_back((it->vertex(i))->info());
        if ((cc.z() + cr) > right_edge[2])
          for (i = 0; i < 4; i++) rz.push_back((it->vertex(i))->info());
        if ((cc.x() - cr) < left_edge[0])
          for (i = 0; i < 4; i++) lx.push_back((it->vertex(i))->info());
        if ((cc.y() - cr) < left_edge[1])
          for (i = 0; i < 4; i++) ly.push_back((it->vertex(i))->info());
        if ((cc.z() - cr) < left_edge[2])
          for (i = 0; i < 4; i++) lz.push_back((it->vertex(i))->info());
      }
    }

    std::sort( alln.begin(), alln.end() );
    std::sort( lx.begin(), lx.end() );
    std::sort( ly.begin(), ly.end() );
    std::sort( lz.begin(), lz.end() );
    std::sort( rx.begin(), rx.end() );
    std::sort( ry.begin(), ry.end() );
    std::sort( rz.begin(), rz.end() );
    alln.erase( std::unique( alln.begin(), alln.end() ), alln.end() );
    lx.erase( std::unique( lx.begin(), lx.end() ), lx.end() );
    ly.erase( std::unique( ly.begin(), ly.end() ), ly.end() );
    lz.erase( std::unique( lz.begin(), lz.end() ), lz.end() );
    rx.erase( std::unique( rx.begin(), rx.end() ), rx.end() );
    ry.erase( std::unique( ry.begin(), ry.end() ), ry.end() );
    rz.erase( std::unique( rz.begin(), rz.end() ), rz.end() );
  }
};

