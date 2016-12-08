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
#include <CGAL/config.h>
#if (CGAL_VERSION_NR < 1030501000)
#define VALID 0
#include "dummy_CGAL.hpp"
#else
#define VALID 1
#if (CGAL_VERSION_NR >= 1040401000)
#include <CGAL/Delaunay_triangulation_cell_base_with_circumcenter_3.h>
#else
#include <CGAL/Triangulation_cell_base_with_circumcenter_3.h>
#endif
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Periodic_3_triangulation_ds_vertex_base_3.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Unique_hash_map.h>
#endif
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel           K3p;
typedef CGAL::Periodic_3_Delaunay_triangulation_traits_3<K3p>         Gt3p;
typedef CGAL::Periodic_3_triangulation_ds_vertex_base_3<>             VbDS3p;
typedef CGAL::Triangulation_vertex_base_3<Gt3p,VbDS3p>                Vbb3p;
typedef CGAL::Periodic_3_triangulation_ds_cell_base_3<>               CbDS3p;
typedef CGAL::Triangulation_cell_base_3<Gt3p,CbDS3p>                  Cbb3p;
#if (CGAL_VERSION_NR >= 1040401000)
typedef CGAL::Delaunay_triangulation_cell_base_with_circumcenter_3<Gt3p,Cbb3p> Cb3p;
#else
typedef CGAL::Triangulation_cell_base_with_circumcenter_3<Gt3p,Cbb3p>          Cb3p;
#endif


template <typename Info_>
class PeriodicDelaunay_with_info_3
{
 public:
  typedef CGAL::Triangulation_vertex_base_with_info_3<Info_, Gt3p, Vbb3p> Vb;
  typedef CGAL::Triangulation_data_structure_3<Vb, Cb3p>                  Tds;
  typedef CGAL::Periodic_3_Delaunay_triangulation_3<Gt3p, Tds>            Delaunay;
  typedef Info_ Info;
  typedef typename Delaunay::Point                     Point;
  typedef typename Delaunay::Segment                   Segment;
  typedef typename Delaunay::Triangle                  Triangle;
  typedef typename Delaunay::Tetrahedron               Tetrahedron;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Edge                      Edge_handle;  // not really a handle, just for disambiguation
  typedef typename Delaunay::Facet                     Facet_handle; // not really a handle, just for disambiguation
  typedef typename Delaunay::Cell_handle               Cell_handle;
  typedef typename Delaunay::Facet_circulator          Facet_circulator;
  typedef typename Delaunay::Cell_circulator           Cell_circulator;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Edge_iterator             Edge_iterator;
  typedef typename Delaunay::Facet_iterator            Facet_iterator;
  typedef typename Delaunay::Cell_iterator             Cell_iterator;
  typedef typename Delaunay::Locate_type               Locate_type;
  typedef typename Delaunay::Iso_cuboid                Iso_cuboid;
  typedef typename Delaunay::Covering_sheets           Covering_sheets;
  typedef typename Delaunay::Offset                    Offset;
  typedef typename Delaunay::Periodic_point            Periodic_point;
  typedef typename Delaunay::Periodic_segment          Periodic_segment;
  typedef typename Delaunay::Periodic_triangle         Periodic_triangle;
  typedef typename Delaunay::Periodic_tetrahedron      Periodic_tetrahedron;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Cell_handle,int>    Cell_hash;
  Delaunay T;
  bool updated = false;
  PeriodicDelaunay_with_info_3(const double *domain = NULL) {
    if (domain != NULL)
      set_domain(domain);
  }
  PeriodicDelaunay_with_info_3(double *pts, Info *val, uint32_t n,
                               const double *domain = NULL) {
    if (domain != NULL)
      set_domain(domain);
    insert(pts, val, n);
  }
  bool is_valid() const { return T.is_valid(); }
  void num_sheets(int32_t* ns_out) const {
    Covering_sheets ns_dim = T.number_of_sheets();
    int i;
    for (i = 0; i < 3; i++)
      ns_out[i] = (int32_t)(ns_dim[i]);
  }
  uint32_t num_sheets_total() const {
    Covering_sheets ns_dim = T.number_of_sheets();
    uint32_t ns = 1;
    int i;
    for (i = 0; i < 3; i++)
      ns = ns*ns_dim[i];
    return ns;
  }
  uint32_t num_finite_verts() const { return static_cast<uint32_t>(T.number_of_vertices()); }
  uint32_t num_finite_edges() const { return static_cast<uint32_t>(T.number_of_edges()); }
  uint32_t num_finite_facets() const { return static_cast<uint32_t>(T.number_of_facets()); }
  uint32_t num_finite_cells() const { return static_cast<uint32_t>(T.number_of_cells()); }
  uint32_t num_infinite_verts() const { return static_cast<uint32_t>(0); }
  uint32_t num_infinite_edges() const { return static_cast<uint32_t>(0); }
  uint32_t num_infinite_facets() const { return static_cast<uint32_t>(0); }
  uint32_t num_infinite_cells() const { return static_cast<uint32_t>(0); }
  uint32_t num_verts() const { return (num_finite_verts() + num_infinite_verts()); }
  uint32_t num_edges() const { return (num_finite_edges() + num_infinite_edges()); }
  uint32_t num_facets() const { return (num_finite_facets() + num_infinite_facets()); }
  uint32_t num_cells() const { return (num_finite_cells() + num_infinite_cells()); }
  uint32_t num_stored_verts() const { return static_cast<uint32_t>(T.number_of_stored_vertices()); }
  uint32_t num_stored_edges() const { return static_cast<uint32_t>(T.number_of_stored_edges()); }
  uint32_t num_stored_facets() const { return static_cast<uint32_t>(T.number_of_stored_facets()); }
  uint32_t num_stored_cells() const { return static_cast<uint32_t>(T.number_of_stored_cells()); }

  bool is_equal(const PeriodicDelaunay_with_info_3<Info> other) const {
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

  void set_domain(const double *domain) {
    Iso_cuboid dr(domain[0], domain[1], domain[2], 
		  domain[3], domain[4], domain[5]);
    T.set_domain(dr);
  }
  void insert(double *pts, Info *val, uint32_t n)
  {
    updated = true;
    uint32_t d, d3;
    std::size_t i;
    Vertex_handle v;
    Point p;
    for (d = 0; d < n; d++) {
      d3 = 3*d;
      p = Point(pts[d3],pts[d3+1],pts[d3+2]);
      v = T.insert(p);
      v->info() = val[d];
      // std::cout << p << val[d] << std::endl;
      std::vector<Vertex_handle> dups = T.periodic_copies(v);
      for (i = 0; i < dups.size(); i++)
	dups[i]->info() = val[d];
    }
  }
  void remove(Vertex v) { updated = true; T.remove(v._x); }
  void clear() { updated = true; T.clear(); }

  Vertex move(Vertex v, double *pos) {
    updated = true;
    Point p = Point(pos[0], pos[1], pos[2]);
    // Not implemented in CGAL as of 4.9. Implemeted her as stop gap.
    // return Vertex(T.move(v._x, p));
    T.remove(v._x);
    v._x = T.insert(p);
    return v;
  }
  Vertex move_if_no_collision(Vertex v, double *pos) {
    updated = true;
    Point p = Point(pos[0], pos[1], pos[2]);
    // Not implemented in CGAL as of 4.9. Implemeted her as stop gap.
    // return Vertex(T.move_if_no_collision(v._x, p));
    int li, lj;
    Locate_type lt = Locate_type(0);
    Cell_handle c = T.locate(p, lt, li, lj, v._x->cell());
    if (lt == 0)
      return Vertex(c->vertex(li));
    else {
      T.remove(v._x);
      v._x = T.insert(p, lt, c, li, lj);
      return v;
    }
      
  }

  Vertex get_vertex(Info index) const {
    Vertex_iterator it = T.vertices_begin();
    for ( ; it != T.vertices_end(); it++) {
      if (it->info() == index)
        return Vertex(T.get_original_vertex(it));
    }
    return Vertex(Vertex_handle());
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

  bool is_unique(Vertex v) const {
    return (!has_offset(v));
  }
  bool is_unique(Edge e) const {
    if (!has_offset(e))
      return true;
    int i, j;
    Offset o;
    Covering_sheets cover = T.number_of_sheets();
    int off[3] = {1, 1, 1};
    Offset eoff[2];
    get_edge_offsets(e, eoff[0], eoff[1]);
    if ((cover[0] != 1) && (cover[1] != 1) && (cover[2] != 1)) {
      for (i = 0; i < 2; i++) {
	o = eoff[i];
	for (j = 0; j < 3; j++) {
	  if (o[j] > 1)
	    return false;
	  off[j] = off[j] & o[j];
	}
      }
    }
    for (j = 0; j < 3; j++)
      if (off[j] != 0)
	return false;
    return true;
  }
  bool is_unique(Facet f) const {
    if (!has_offset(f))
      return true;
    int i, j;
    Offset o;
    Covering_sheets cover = T.number_of_sheets();
    int off[3] = {1, 1, 1};
    Offset eoff[3];
    get_edge_offsets(f, eoff[0], eoff[1], eoff[2]);
    if ((cover[0] != 1) && (cover[1] != 1) && (cover[2] != 1)) {
      for (i = 0; i < 3; i++) {
	o = eoff[i];
	for (j = 0; j < 3; j++) {
	  if (o[j] > 1)
	    return false;
	  off[j] = off[j] & o[j];
	}
      }
    }
    for (j = 0; j < 3; j++)
      if (off[j] != 0)
	return false;
    return true;
  }
  // bool is_unique(Facet f) const {
  //   if (!has_offset(f))
  //     return true;
  //   int i;
  //   Offset o;
  //   Covering_sheets cover = T.number_of_sheets();
  //   for (i = 0; i < 3; i++) {
  //     o = get_offset(f, i);
  //     if (((o.x()%cover[0]) == 0) and ((o.y()%cover[1]) == 0) and ((o.z()%cover[2]) == 0))
  // 	return true;
  //   }
  //   return false;
  // }
  bool is_unique(Cell c) const {
    if (!has_offset(c))
      return true;
    int i, j;
    Offset o;
    Covering_sheets cover = T.number_of_sheets();
    int off[3] = {1, 1, 1};
    Offset eoff[4];
    get_edge_offsets(c, eoff[0], eoff[1], eoff[2], eoff[3]);
    if ((cover[0] != 1) && (cover[1] != 1) && (cover[2] != 1)) {
      for (i = 0; i < 4; i++) {
	o = eoff[i];
	for (j = 0; j < 3; j++) {
	  if (o[j] > 1)
	    return false;
	  off[j] = off[j] & o[j];
	}
      }
    }
    for (j = 0; j < 3; j++)
      if (off[j] != 0)
	return false;
    return true;
  }
  // bool is_unique(Cell c) const {
  //   if (!has_offset(c))
  //     return true;
  //   int i;
  //   Offset o;
  //   Covering_sheets cover = T.number_of_sheets();
  //   for (i = 0; i < 4; i++) {
  //     o = get_offset(c, i);
  //     if (((o.x()%cover[0]) == 0) and ((o.y()%cover[1]) == 0) and ((o.z()%cover[2]) == 0))
  // 	return true;
  //   }
  //   return false;
  // }
  bool spans_wrap(Edge e) const {
    int i, j;
    int mins[3] = {9,9,9};
    int maxs[3] = {0,0,0};
    Offset o;
    Covering_sheets cover = T.number_of_sheets();
    for (i = 0; i < 2; i++) {
      o = get_offset(e, i);
      for (j = 0; j < 3; j++) {
	mins[j] = std::min(mins[j], o[j]%cover[j]);
	maxs[j] = std::max(maxs[j], o[j]%cover[j]);
      }
    }
    for (j = 0; j < 3; j++) {
      if ((maxs[j]-mins[j]) > 1)
	return true;
    }
    return false;
  }
  bool spans_wrap(Facet f) const {
    int i, j;
    int mins[3] = {9,9,9};
    int maxs[3] = {0,0,0};
    Offset o;
    Covering_sheets cover = T.number_of_sheets();
    for (i = 0; i < 3; i++) {
      o = get_offset(f, i);
      for (j = 0; j < 3; j++) {
	mins[j] = std::min(mins[j], o[j]%cover[j]);
	maxs[j] = std::max(maxs[j], o[j]%cover[j]);
      }
    }
    for (j = 0; j < 3; j++) {
      if ((maxs[j]-mins[j]) > 1)
	return true;
    }
    return false;
  }
  bool spans_wrap(Cell c) const {
    int i, j;
    int mins[3] = {9,9,9};
    int maxs[3] = {0,0,0};
    Offset o;
    Covering_sheets cover = T.number_of_sheets();
    for (i = 0; i < 4; i++) {
      o = get_offset(c, i);
      for (j = 0; j < 3; j++) {
	mins[j] = std::min(mins[j], o[j]%cover[j]);
	maxs[j] = std::max(maxs[j], o[j]%cover[j]);
      }
    }
    for (j = 0; j < 3; j++) {
      if ((maxs[j]-mins[j]) > 1)
	return true;
    }
    return false;
  }
  void get_edge_offsets(Edge e, Offset &off0, Offset &off1) const {
    Offset cell_off0 = T.int_to_off(e._x.first->offset(e._x.second));
    Offset cell_off1 = T.int_to_off(e._x.first->offset(e._x.third));
    Offset diff_off((cell_off0.x()==1 && cell_off1.x()==1)?-1:0,
		    (cell_off0.y()==1 && cell_off1.y()==1)?-1:0,
		    (cell_off0.z()==1 && cell_off1.z()==1)?-1:0);
    off0 = T.combine_offsets(T.get_offset(e._x.first,e._x.second),
			     diff_off);
    off1 = T.combine_offsets(T.get_offset(e._x.first,e._x.third),
			     diff_off);
  }
  void get_edge_offsets(Facet f, Offset &off0, Offset &off1, Offset &off2) const {
    Offset cell_off0 = T.int_to_off(f._x.first->offset((f._x.second+1)&3));
    Offset cell_off1 = T.int_to_off(f._x.first->offset((f._x.second+2)&3));
    Offset cell_off2 = T.int_to_off(f._x.first->offset((f._x.second+3)&3));
    Offset diff_off((cell_off0.x() == 1 
		     && cell_off1.x() == 1 
		     && cell_off2.x() == 1)?-1:0,
		    (cell_off0.y() == 1 
		     && cell_off1.y() == 1
		     && cell_off2.y() == 1)?-1:0,
		    (cell_off0.z() == 1 
		     && cell_off1.z() == 1
		     && cell_off2.z() == 1)?-1:0);
    off0 = T.combine_offsets(T.get_offset(f._x.first,
					  (f._x.second+1)&3),
			     diff_off);
    off1 = T.combine_offsets(T.get_offset(f._x.first,
					  (f._x.second+2)&3),
			     diff_off);
    off2 = T.combine_offsets(T.get_offset(f._x.first,
					  (f._x.second+3)&3),
			     diff_off);
  }
  void get_edge_offsets(Cell c, Offset &off0, Offset &off1,
			Offset &off2, Offset &off3) const {
    // Offset cell_off0 = T.int_to_off(c._x.offset(0));
    // Offset cell_off1 = T.int_to_off(c._x.offset(1));
    // Offset cell_off2 = T.int_to_off(c._x.offset(2));
    // Offset cell_off3 = T.int_to_off(c._x.offset(3));
    Offset cell_off0 = T.get_offset(c._x,0);
    Offset cell_off1 = T.get_offset(c._x,1);
    Offset cell_off2 = T.get_offset(c._x,2);
    Offset cell_off3 = T.get_offset(c._x,3);
    Offset diff_off((cell_off0.x() == 1 
		     && cell_off1.x() == 1 
		     && cell_off2.x() == 1
		     && cell_off3.x() == 1)?-1:0,
		    (cell_off0.y() == 1 
		     && cell_off1.y() == 1
		     && cell_off2.y() == 1
		     && cell_off3.y() == 1)?-1:0,
		    (cell_off0.z() == 1 
		     && cell_off1.z() == 1
		     && cell_off2.z() == 1
		     && cell_off3.z() == 1)?-1:0);
    off0 = T.combine_offsets(T.get_offset(c._x,0), diff_off);
    off1 = T.combine_offsets(T.get_offset(c._x,1), diff_off);
    off2 = T.combine_offsets(T.get_offset(c._x,2), diff_off);
    off3 = T.combine_offsets(T.get_offset(c._x,3), diff_off);
  }
  Offset get_offset(Vertex v) const {
    return T.get_offset(v._x);
  }
  Offset get_offset(Edge e, int i) const {
    return T.periodic_segment(e._x)[i%2].second;
  }
  Offset get_offset(Facet f, int i) const {
    return T.periodic_triangle(f._x)[i%3].second;
  }
  Offset get_offset(Cell c, int i) const {
    return T.periodic_tetrahedron(c._x)[i%4].second;
  }
  bool has_offset(Vertex v) const {
    Offset o = get_offset(v);
    if ((o.x() > 0) or (o.y() > 0) or (o.z() > 0))
      return true;
    else
      return false;
  }
  bool has_offset(Edge e) const {
    Offset o ;
    for (int i = 0; i < 2; i++) {
      o = get_offset(e, i);
      if ((o.x() > 0) or (o.y() > 0) or (o.z() > 0))
	return true;
    }
    return false;
  }
  bool has_offset(Facet f) const {
    Offset o ;
    for (int i = 0; i < 3; i++) {
      o = get_offset(f, i);
      if ((o.x() > 0) or (o.y() > 0) or (o.z() > 0))
	return true;
    }
    return false;
  }
  bool has_offset(Cell c) const {
    Offset o ;
    for (int i = 0; i < 4; i++) {
      o = get_offset(c, i);
      if ((o.x() > 0) or (o.y() > 0) or (o.z() > 0))
	return true;
    }
    return false;
  }
  void point(Vertex v, double* pos) const {
    Point p = T.point(T.periodic_point(v._x));
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void point(Edge e, int i, double* pos) const {
    Point p = T.segment(T.periodic_segment(e._x)).vertex(i%2);
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void point(Facet f, int i, double* pos) const {
    Point p = T.triangle(T.periodic_triangle(f._x)).vertex(i%3);
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void point(Cell c, int i, double* pos) const {
    Point p = T.tetrahedron(T.periodic_tetrahedron(c._x)).vertex(i%4);
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void periodic_point(Vertex v, double* pos) const {
    Point p = v._x->point();
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void periodic_point(Edge e, int i, double* pos) const {
    Point p = T.periodic_segment(e._x)[i%2].first;
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void periodic_point(Facet f, int i, double* pos) const {
    Point p = T.periodic_triangle(f._x)[i%3].first;
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void periodic_point(Cell c, int i, double* pos) const {
    Point p = T.periodic_tetrahedron(c._x)[i%4].first;
    pos[0] = p.x();
    pos[1] = p.y();
    pos[2] = p.z();
  }
  void periodic_offset(Vertex v, int32_t* off) const {
    Offset o = get_offset(v);
    off[0] = o.x();
    off[1] = o.y();
    off[2] = o.z();
  }
  void periodic_offset(Edge e, int i, int32_t* off) const {
    Offset o = get_offset(e, i);
    off[0] = o.x();
    off[1] = o.y();
    off[2] = o.z();
  }
  void periodic_offset(Facet f, int i, int32_t* off) const {
    Offset o = get_offset(f, i);
    off[0] = o.x();
    off[1] = o.y();
    off[2] = o.z();
  }
  void periodic_offset(Cell c, int i, int32_t* off) const {
    Offset o = get_offset(c, i);
    off[0] = o.x();
    off[1] = o.y();
    off[2] = o.z();
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
    void offset(int32_t* out) const {
      Offset o = _x->offset();
      out[0] = o.x();
      out[1] = o.y();
      out[2] = o.z();
    }
    Info info() { return _x->info(); }
    Cell cell() { return Cell(_x->cell()); }
    void set_cell(Cell c) { _x->set_cell(c._x); }
    void set_point(double *x) {
      Point p = Point(x[0], x[1], x[2]);
      _x->set_point(p);
    }
    void set_offset(int32_t* x) {
      Offset o = Offset(x[0], x[1], x[2]);
      _x->set_offset(o);
    }
  };

  // Edge construct
  class All_edges_iter {
  public:
    Edge_iterator _x = Edge_iterator();
    All_edges_iter() { _x = Edge_iterator(); }
    All_edges_iter(Edge_iterator x) { _x = x; }
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
    Edge(Edge_iterator x) { _x = static_cast<Edge_handle>(*x); }
    Edge(All_edges_iter x) { _x = static_cast<Edge_handle>(*(x._x)); }
    Edge(Cell x, int i1, int i2) { _x = Edge_handle(x._x, i1%4, i2%4); }
    Cell cell() const { return Cell(_x.first); }
    Vertex vertex(int i) {
      if ((i % 2) == 0)
	return v1();
      else
	return v2();
    }
    int ind(int i) {
      if ((i % 2) == 0)
	return ind1();
      else
	return ind2();
    }
    int ind1() const { return _x.second; }
    int ind2() const { return _x.third; }
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
  All_facets_iter all_facets_begin() { return All_facets_iter(T.all_facets_begin()); }
  All_facets_iter all_facets_end() { return All_facets_iter(T.all_facets_end()); }

  class Facet {
  public:
    Facet_handle _x = Facet_handle();
    Facet() {}
    Facet(Facet_handle x) { _x = x; }
    Facet(Facet_iterator x) { _x = static_cast<Facet_handle>(*x); }
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
  bool is_infinite(Vertex x) const { return false; }
  bool is_infinite(Edge x) const { return false; }
  bool is_infinite(Facet x) const { return false; }
  bool is_infinite(Cell x) const { return false; }
  bool is_infinite(All_verts_iter x) const { return false; }
  bool is_infinite(All_edges_iter x) const { return false; }
  bool is_infinite(All_facets_iter x) const { return false; }
  bool is_infinite(All_cells_iter x) const { return false; }

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
    Point p = T.point(T.periodic_circumcenter(x._x));
    // Point p = x._x->circumcenter();
    out[0] = p.x();
    out[1] = p.y();
    out[2] = p.z();
  }

  void periodic_circumcenter(Cell x, double* out) const {
    Point p = T.periodic_circumcenter(x._x).first;
    out[0] = p.x();
    out[1] = p.y();
    out[2] = p.z();
  }

  double dual_volume(const Vertex v) const {
    std::list<Edge_handle> edges;
    T.incident_edges(v._x, std::back_inserter(edges));

    Point orig = T.point(T.periodic_point(v._x));
    double vol = 0.0;
    for (typename std::list<Edge_handle>::iterator eit = edges.begin() ;
         eit != edges.end() ; ++eit) {

      Facet_circulator fstart = T.incident_facets(*eit);
      Facet_circulator fcit = fstart;
      std::vector<Point> pts;
      do {
	Point dual_orig = T.point(T.periodic_circumcenter(fcit->first));
        // Point dual_orig = fcit->first->circumcenter();
        pts.push_back(dual_orig);
        ++fcit;
      } while (fcit != fstart);

      for (uint32_t i=1 ; i<pts.size()-1 ; i++)
        vol += Tetrahedron(orig,pts[0],pts[i],pts[i+1]).volume();
    }
    return vol;
  }

  void dual_volumes(double *vols) const {
    for (Vertex_iterator it = T.vertices_begin(); it != T.vertices_end(); it++) {
      vols[it->info()] = dual_volume(Vertex(it));
    }    
  }

  double length(const Edge e) const {
    Segment s = T.segment(T.periodic_segment(e._x));
    double out = std::sqrt(s.squared_length());
    // Vertex_handle v1 = e.v1()._x;
    // Vertex_handle v2 = e.v2()._x;
    // Point p1 = v1->point();
    // Point p2 = v2->point();
    // double out = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, p2)));
    return out;
  }

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
    if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
      return 1;
    else {
      Point p = Point(pos[0], pos[1], pos[2]);
      Locate_type lt_out = Locate_type(0);    
      int out = -(int)T.side_of_cell(p, c._x, lt_out, li, lj);
      lt = (int)lt_out;
      return out;
    }
  }
  // Currently segfaults inside CGAL function call
  // int side_of_circle(const Facet f, const double* pos) const {
  //   if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
  //     return 1;
  //   else {
  //     Point p = Point(pos[0], pos[1], pos[2]);
  //     return -(int)(T.side_of_circle(f.cell()._x, f.ind(), p));
  //     // return (int)(-T.side_of_circle(f._x, p));
  //   }
  // }
  int side_of_sphere(const Cell c, const double* pos) const {
    if (std::isinf(pos[0]) || std::isinf(pos[1]) || std::isinf(pos[2])) 
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
    std::streambuf* oldCoutStreamBuf = std::cout.rdbuf();
    std::ostringstream newCoutStream;
    std::cout.rdbuf( newCoutStream.rdbuf() );
    os << T;
    // T.save(os);
    std::cout.rdbuf( oldCoutStreamBuf );
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
    std::streambuf* oldCoutStreamBuf = std::cout.rdbuf();
    std::ostringstream newCoutStream;
    std::cout.rdbuf( newCoutStream.rdbuf() );
    is >> T;
    // T.load(is);
    std::cout.rdbuf( oldCoutStreamBuf );
  }

  template <typename I>
  I serialize(I &n, I &m, int32_t &d, double* domain, int32_t* cover,
              double* vert_pos, Info* vert_info,
              I* cells, I* neighbors, int32_t* offsets) const
  {
    I idx_inf = std::numeric_limits<I>::max();
    Delaunay T1 = T;
    T1.convert_to_1_sheeted_covering();
    int j;

    // Header
    Iso_cuboid dom_rect = T1.domain();
    Covering_sheets ns_dim = T.number_of_sheets();
    n = static_cast<int>(T1.number_of_vertices());
    m = static_cast<int>(T1.tds().number_of_cells());
    d = static_cast<int>(T1.dimension());
    domain[0] = dom_rect.xmin();
    domain[1] = dom_rect.ymin();
    domain[2] = dom_rect.zmin();
    domain[3] = dom_rect.xmax();
    domain[4] = dom_rect.ymax();
    domain[5] = dom_rect.zmax();
    for (j = 0; j < d; j++)
      cover[j] = (int32_t)(ns_dim[j]);
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Vertex_hash V;
    Cell_hash C;
    Vertex_handle vit;
    int inum = 0;
    
    // vertices
    inum = 0;
    for (Vertex_iterator vit = T1.vertices_begin(); vit != T1.vertices_end(); vit++) {
      vert_pos[d*inum + 0] = static_cast<double>(vit->point().x());
      vert_pos[d*inum + 1] = static_cast<double>(vit->point().y());
      vert_pos[d*inum + 2] = static_cast<double>(vit->point().z());
      vert_info[inum] = vit->info();
      V[vit] = inum++;
    }
    
    // vertices of the cells
    inum = 0;
    for (Cell_iterator ib = T.tds().cells_begin(); 
	 ib != T.tds().cells_end(); ++ib) {
      for (j = 0; j < dim ; ++j) {
	vit = ib->vertex(j);
	cells[dim*inum + j] = V[vit];
      }
      C[ib] = inum++;
    }
  
    // neighbor pointers of the cells
    inum = 0;
    for( Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      for (j = 0; j < d+1; ++j){
	neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
      }
      inum++;
    }

    // offsets
    inum = 0;
    for( Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      for (j = 0; j < d+1; ++j){
	offsets[(d+1)*inum + j] = it->offset(j);
      }
      inum++;
    }

    return idx_inf;
  }

  template <typename I>
  Info serialize_idxinfo(I &n, I &m, int32_t &d, double* domain, int32_t* cover,
			 Info* cells, I* neighbors, int32_t *offsets) const
  {
    Info idx_inf = std::numeric_limits<Info>::max();
    int j;
    Delaunay T1 = T;
    T1.convert_to_1_sheeted_covering();

    // Header
    Iso_cuboid dom_rect = T1.domain();
    Covering_sheets ns_dim = T.number_of_sheets();
    n = static_cast<int>(T1.number_of_vertices());
    m = static_cast<int>(T1.tds().number_of_cells());
    d = static_cast<int>(T1.dimension());
    domain[0] = dom_rect.xmin();
    domain[1] = dom_rect.ymin();
    domain[2] = dom_rect.zmin();
    domain[3] = dom_rect.xmax();
    domain[4] = dom_rect.ymax();
    domain[5] = dom_rect.zmax();
    for (j = 0; j < d; j++)
      cover[j] = (int32_t)(ns_dim[j]);
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Cell_hash C;
    Vertex_handle vit;
    int inum = 0;
    
    // vertices of the cells
    inum = 0;
    for (Cell_iterator ib = T.tds().cells_begin(); 
	 ib != T.tds().cells_end(); ++ib) {
      for (j = 0; j < dim ; ++j) {
	vit = ib->vertex(j);
	cells[dim*inum + j] = vit->info();
      }
      C[ib] = inum++;
    }
  
    // neighbor pointers of the cells
    inum = 0;
    for (Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      for (j = 0; j < d+1; ++j){
	neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
      }
      inum++;
    }

    // offsets
    inum = 0;
    for (Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      for (j = 0; j < d+1; ++j){
	offsets[(d+1)*inum + j] = it->offset(j);
      }
      inum++;
    }

    return idx_inf;
  }

  template <typename I>
  I serialize_info2idx(I &n, I &m, int32_t &d,
		       double* domain, int32_t* cover,
		       I* cells, I* neighbors, int32_t* offsets,
		       Info max_info, I* idx) const
  {
    I idx_inf = std::numeric_limits<I>::max();
    int j;
    Delaunay T1 = T;
    T1.convert_to_1_sheeted_covering();

    // Header
    Iso_cuboid dom_rect = T1.domain();
    Covering_sheets ns_dim = T.number_of_sheets();
    n = static_cast<I>(T1.number_of_vertices());
    d = static_cast<int>(T1.dimension());
    domain[0] = dom_rect.xmin();
    domain[1] = dom_rect.ymin();
    domain[2] = dom_rect.zmin();
    domain[3] = dom_rect.xmax();
    domain[4] = dom_rect.ymax();
    domain[5] = dom_rect.zmax();
    for (j = 0; j < d; j++)
      cover[j] = (int32_t)(ns_dim[j]);
    int dim = (d == -1 ? 1 :  d + 1);
    if ((n == 0) || (m == 0)) {
      return idx_inf;
    }

    Cell_hash C;
    Vertex_handle vit;
    int inum, inum_tot;
    
    // vertices of the cells
    bool *include_cell = (bool*)malloc(m*sizeof(bool));
    inum = 0, inum_tot = 0;
    for (Cell_iterator ib = T.tds().cells_begin(); 
	 ib != T.tds().cells_end(); ++ib) {
      include_cell[inum_tot] = false;
      for (j = 0; j < dim ; ++j) {
        vit = ib->vertex(j);
	if (vit->info() < max_info) {
          include_cell[inum_tot] = true;
	  break;
        }
      }
      if (include_cell[inum_tot]) {
	for (j = 0; j < dim ; ++j) {
	  vit = ib->vertex(j);
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
    for (Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      if (include_cell[inum_tot]) {
	for (j = 0; j < d+1; ++j){
	  neighbors[(d+1)*inum + j] = C[it->neighbor(j)];
	}
	inum++;
      }
      inum_tot++;
    }

    // offsets
    inum = 0, inum_tot = 0;
    for (Cell_iterator it = T.tds().cells_begin();
	 it != T.tds().cells_end(); ++it) {
      if (include_cell[inum_tot]) {
	for (j = 0; j < d+1; ++j){
	  offsets[(d+1)*inum + j] = it->offset(j);
	}
	inum++;
      }
      inum_tot++;
    }

    free(include_cell);
    return idx_inf;
  }

  template <typename I>
  void deserialize(I n, I m, int32_t d, double* domain, int32_t *cover,
                   double* vert_pos, Info* vert_info,
                   I* cells, I* neighbors, int32_t* offsets, I idx_inf)
  {
    updated = true;

    T.clear();
 
    if (n==0) {
      return;
    }

    Iso_cuboid dom_rect(domain[0],domain[1],domain[2],
			domain[3],domain[4],domain[5]);
    T.convert_to_1_sheeted_covering();
    T.set_domain(dom_rect);
    T.tds().set_dimension(d);

    int32_t ns = cover[0]*cover[1];

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    I i;
    int j;

    // read vertices
    for (i = 0; i < n; ++i) {
      V[i] = T.tds().create_vertex();
      V[i]->point() = Point(vert_pos[d*i], vert_pos[d*i + 1], vert_pos[d*i + 2]);
      V[i]->info() = vert_info[i];
    }

    // First cell
    i = 0;
    if (T.cells_begin() != T.cells_end()) {
      C[i] = T.cells_begin();
      for (j = 0; j < dim; ++j) {
        index = cells[dim*i + j];
        v = V[index];
        C[i]->set_vertex(j, v);
        v->set_cell(C[i]);
      }
      i++;
    }

    // Creation of the cells
    for ( ; i < m; ++i) {
      C[i] = T.tds().create_cell() ;
      for (j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
        v = V[index];
        C[i]->set_vertex(j, v);
        v->set_cell(C[i]);
      }
    }

    // Setting the neighbor pointers
    for (i = 0; i < m; ++i) {
      for(j = 0; j < d+1; ++j){
        index = neighbors[(d+1)*i + j];
        C[i]->set_neighbor(j, C[index]);
      }
    }

    // Setting the offset of vertices
    for (i = 0; i < m; ++i) {
      T.set_offsets(C[i],
                    offsets[(d+1)*i],
                    offsets[(d+1)*i + 1],
                    offsets[(d+1)*i + 2],
		    offsets[(d+1)*i + 3]);
    }

    // Restore to 27 sheet covering if necessary
    if (ns == 27) {
      T.convert_to_27_sheeted_covering();
    }

  }

  template <typename I>
  void deserialize_idxinfo(I n, I m, int32_t d, 
			   double* domain, int32_t *cover, double* vert_pos, 
			   I* cells, I* neighbors, int32_t* offsets, I idx_inf)
  {
    updated = true;

    T.clear();
 
    if (n==0) {
      return;
    }

    Iso_cuboid dom_rect(domain[0],domain[1],domain[2],
			domain[3],domain[4],domain[5]);
    T.convert_to_1_sheeted_covering();
    T.set_domain(dom_rect);
    T.tds().set_dimension(d);

    int32_t ns = cover[0]*cover[1];

    std::vector<Vertex_handle> V(n+1);
    std::vector<Cell_handle> C(m);
    Vertex_handle v;
    I index;
    int dim = (d == -1 ? 1 : d + 1);
    I i;
    int j;

    // read vertices
    for (i = 0; i < n; ++i) {
      V[i] = T.tds().create_vertex();
      V[i]->point() = Point(vert_pos[d*i], vert_pos[d*i + 1], vert_pos[d*i + 2]);
      V[i]->info() = (Info)(i);
    }

    // First cell
    i = 0;
    if (T.cells_begin() != T.cells_end()) {
      C[i] = T.cells_begin();
      for (j = 0; j < dim; ++j) {
        index = cells[dim*i + j];
        v = V[index];
        C[i]->set_vertex(j, v);
        v->set_cell(C[i]);
      }
      i++;
    }

    // Creation of the cells
    for (i = 0; i < m; ++i) {
      C[i] = T.tds().create_cell() ;
      for (j = 0; j < dim ; ++j){
        index = cells[dim*i + j];
	v = V[index];
        C[i]->set_vertex(j, v);
        v->set_cell(C[i]);
      }
    }

    // Setting the neighbor pointers
    for (i = 0; i < m; ++i) {
      for (j = 0; j < d+1; ++j){
        index = neighbors[(d+1)*i + j];
        C[i]->set_neighbor(j, C[index]);
      }
    }

    // Setting the offsets
    for (i = 0; i < m; ++i) {
      T.set_offsets(C[i],
                    offsets[(d+1)*i],
                    offsets[(d+1)*i + 1],
                    offsets[(d+1)*i + 2],
		    offsets[(d+1)*i + 3]);
    }

    // Restore to 27 sheet covering if necessary
    if (ns == 27) {
      T.convert_to_27_sheeted_covering();
    }

  }

  void info_ordered_vertices(double* pos) const {
    Info i;
    Point p;
    for (Vertex_iterator it = T.vertices_begin(); it != T.vertices_end(); it++) {
      i = it->info();
      p = it->point();
      pos[3*i + 0] = p.x();
      pos[3*i + 1] = p.y();
      pos[3*i + 2] = p.z();
    }
  }

  void vertex_info(Info* verts) const {
    int i = 0;
    for (Vertex_iterator it = T.vertices_begin(); it != T.vertices_end(); it++) {
      verts[i] = it->info();
      i++;
    }
  }
  
  void edge_info(Info* edges) const {
    int i = 0;
    Info i1, i2;
    Vertex_handle v1, v2;
    for (Edge_iterator it = T.edges_begin(); it != T.edges_end(); it++) {
      if (is_unique(Edge(it))) {
	v1 = it->first->vertex(it->second);
	v2 = it->first->vertex(it->third);
	i1 = v1->info();
	i2 = v2->info();
	edges[2*i + 0] = i1;
	edges[2*i + 1] = i2;
	i++;
      }
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
    int i;

    for (Cell_iterator it = T.all_cells_begin(); it != T.all_cells_end(); it++) {
      p1 = T.point(T.periodic_point(it->vertex(0)));
      // p1 = it->vertex(0)->point();
      cc = T.point(T.periodic_circumcenter(it));
      // cc = it->circumcenter();
      cr = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, cc)));
      for (b = 0; b < nbox; b++) {
	if (intersect_sph_box(&cc, cr, left_edges + 3*b, right_edges + 3*b))
	  for (i = 0; i < 4; i++) out[b].push_back((it->vertex(i))->info());
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
    double cr;
    int i;

    for (Cell_iterator it = T.all_cells_begin(); it != T.all_cells_end(); it++) {
      p1 = T.point(T.periodic_point(it->vertex(0)));
      // p1 = it->vertex(0)->point();
      cc = T.point(T.periodic_circumcenter(it));
      // cc = it->circumcenter();
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

