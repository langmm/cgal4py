#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_circumcenter_3.h>
#include <CGAL/squared_distance_3.h>
#include <vector>
#include <set>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <CGAL/Unique_hash_map.h>
#include <stdint.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_cell_base_with_circumcenter_3<K>      Cb3;


template <typename Info_>
class Delaunay_with_info_3
// class Delaunay_with_info_3 : public CGAL::Delaunay_triangulation_3<K, CGAL::Triangulation_data_structure_3<CGAL::Triangulation_vertex_base_with_info_3<Info_, K>>>
{
 public:
  typedef CGAL::Delaunay_triangulation_3<K, CGAL::Triangulation_data_structure_3<CGAL::Triangulation_vertex_base_with_info_3<Info_, K>, Cb3>> Delaunay;
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
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Cell_handle,int>    Cell_hash;
  typedef Info_ Info;
  Delaunay T;
  Delaunay_with_info_3() {};
  Delaunay_with_info_3(double *pts, Info *val, uint32_t n) { insert(pts, val, n); }
  uint32_t num_finite_verts() { return static_cast<uint32_t>(T.number_of_vertices()); }
  uint32_t num_finite_edges() { return static_cast<uint32_t>(T.number_of_finite_edges()); }
  uint32_t num_finite_facets() { return static_cast<uint32_t>(T.number_of_finite_facets()); }
  uint32_t num_finite_cells() { return static_cast<uint32_t>(T.number_of_finite_cells()); }
  uint32_t num_infinite_verts() { return 1; }
  uint32_t num_infinite_edges() { return (T.number_of_edges() - T.number_of_finite_edges()); }
  uint32_t num_infinite_facets() { return (T.number_of_facets() - T.number_of_finite_facets()); }
  uint32_t num_infinite_cells() { return (T.number_of_cells() - T.number_of_finite_cells()); }
  uint32_t num_verts() { return (T.number_of_vertices() + num_infinite_verts()); }
  uint32_t num_edges() { return T.number_of_edges(); }
  uint32_t num_facets() { return T.number_of_facets(); }
  uint32_t num_cells() { return T.number_of_cells(); }

  class Vertex;
  class Edge;
  class Facet;
  class Cell;

  void insert(double *pts, Info *val, uint32_t n)
  {
    uint32_t i, j;
    std::vector< std::pair<Point,Info> > points;
    for (i = 0; i < n; i++) {
      j = 3*i;
      points.push_back( std::make_pair( Point(pts[j],pts[j+1],pts[j+2]), val[i]) );
    }
    T.insert( points.begin(),points.end() );
  }
  void remove(Vertex v) { T.remove(v._x); }
  void clear() { T.clear(); }

  Vertex move(Vertex v, double *pos) {
    Point p = Point(pos[0], pos[1], pos[2]);
    return Vertex(T.move(v._x, p));
  }
  Vertex move_if_no_collision(Vertex v, double *pos) {
    Point p = Point(pos[0], pos[1], pos[2]);
    return Vertex(T.move_if_no_collision(v._x, p));
  }

  Vertex get_vertex(Info index) {
    Finite_vertices_iterator it = T.finite_vertices_begin();
    for ( ; it != T.finite_vertices_end(); it++) {
      if (it->info() == index)
        return Vertex(static_cast<Vertex_handle>(it));
    }
    return Vertex(T.infinite_vertex());
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
    Edge(Cell x, int i1, int i2) { _x = Edge_handle(x._x, i1, i2); }
    bool operator==(Edge other) { return (_x == other._x); }
    bool operator!=(Edge other) { return (_x != other._x); }
    Cell cell() { return Cell(_x.first); }
    int ind1() { return _x.second; }
    int ind2() { return _x.third; }
    Vertex vertex(int i) {
      if ((i % 2) == 0)
	return v1();
      else
	return v2();
    }
    Vertex v1() const { 
      Vertex_handle v = _x.first->vertex(_x.second);
      return Vertex(v); 
    }
    Vertex v2() const { 
      Vertex_handle v = _x.first->vertex(_x.third);
      return Vertex(v); 
    }
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
    bool operator==(Facet other) { return (_x == other._x); }
    bool operator!=(Facet other) { return (_x != other._x); }
    Cell cell() { return Cell(_x.first); }
    int ind() { return _x.second; }
    Vertex vertex(int i) { 
      return Vertex(cell().vertex((ind() + 1 + (i%3))%3)); 
    }
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
    bool operator==(Cell other) { return (_x == other._x); }
    bool operator!=(Cell other) { return (_x != other._x); }

    Vertex vertex(int i) { return Vertex(_x->vertex(i)); }
    bool has_vertex(Vertex v) { return _x->has_vertex(v._x); }
    bool has_vertex(Vertex v, int *i) { return _x->has_vertex(v._x, *i); }
    int ind(Vertex v) { return _x->index(v._x); }

    Cell neighbor(int i) { return Cell(_x->neighbor(i)); }
    bool has_neighbor(Cell c) const { return _x->has_neighbor(c._x); }
    bool has_neighbor(Cell c, int *i) const { return _x->has_neighbor(c._x, *i); }
    int ind(Cell c) const { return _x->index(c._x); }

    void set_vertex(int i, Vertex v) { _x->set_vertex(i, v._x); }
    void set_vertices() { _x->set_vertices(); }
    void set_vertices(Vertex v1, Vertex v2, Vertex v3, Vertex v4) {
      _x->set_vertices(v1._x, v2._x, v3._x, v4._x); }
    void set_neighbor(int i, Cell c) { _x->set_neighbor(i, c._x); }
    void set_neighbors() { _x->set_neighbors(); }
    void set_neighbors(Cell c1, Cell c2, Cell c3, Cell c4) {
      _x->set_neighbors(c1._x, c2._x, c3._x, c4._x); 
    }
  };

  // Testing incidence to the infinite vertex
  bool is_infinite(Vertex x) { return T.is_infinite(x._x); }
  bool is_infinite(Edge x) { return T.is_infinite(x._x); }
  bool is_infinite(Facet x) { return T.is_infinite(x._x); }
  bool is_infinite(Cell x) { return T.is_infinite(x._x); }
  bool is_infinite(All_verts_iter x) { return T.is_infinite(x._x); }
  bool is_infinite(All_edges_iter x) { 
    const Edge_iterator e = x._x;
    return T.is_infinite(*e);
  }
  bool is_infinite(All_facets_iter x) {
    const Facet_iterator f = x._x;
    return T.is_infinite(*f);
  }
  bool is_infinite(All_cells_iter x) { return T.is_infinite(x._x); }

  // Parts incident to a vertex
  std::vector<Vertex> incident_vertices(Vertex x) {
    std::vector<Vertex> out;
    T.adjacent_vertices(x._x, wrap_insert_iterator<Vertex,Vertex_handle>(out));
    return out;
  }
  std::vector<Edge> incident_edges(Vertex x) {
    std::vector<Edge> out;
    T.incident_edges(x._x, wrap_insert_iterator<Edge,Edge_handle>(out));
    return out;
  }
  std::vector<Facet> incident_facets(Vertex x) {
    std::vector<Facet> out;
    T.incident_facets(x._x, wrap_insert_iterator<Facet,Facet_handle>(out));
    return out;
  }
  std::vector<Cell> incident_cells(Vertex x) {
    std::vector<Cell> out;
    T.incident_cells(x._x, wrap_insert_iterator<Cell,Cell_handle>(out));
    return out;
  }

  // Parts incident to an edge
  std::vector<Vertex> incident_vertices(Edge x) {
    std::vector<Vertex> out;
    out.push_back(x.v1());
    out.push_back(x.v2());
    return out;
  }
  std::vector<Edge> incident_edges(Edge x) {
    uint32_t i;
    std::vector<Edge> out1, out2, out;
    T.incident_edges(x.v1()._x, wrap_insert_iterator<Edge,Edge_handle>(out1));
    T.incident_edges(x.v2()._x, wrap_insert_iterator<Edge,Edge_handle>(out2));
    for (i = 0; i < out1.size(); i++) {
      if (out1[i] != x)
	out.push_back(out1[i]);
    }
    for (i = 0; i < out2.size(); i++) {
      if (out2[i] != x)
	out.push_back(out2[i]);
    }
    return out;
  }
  std::vector<Facet> incident_facets(Edge x) {
    std::vector<Facet> out;
    Facet_circulator cc = T.incident_facets(x._x), done(cc);
    if (cc == 0)
      return out;
    do {
      out.push_back(Facet(cc));
    } while (++cc != done);
    return out;
  }
  std::vector<Cell> incident_cells(Edge x) {
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
  std::vector<Vertex> incident_vertices(Facet x) {
    std::vector<Vertex> out;
    for (int i = 0; i < 3; i++) {
      out.push_back(x.vertex(i));
    }
    return out;
  }
  std::vector<Edge> incident_edges(Facet x) {
    std::vector<Edge> out;
    int i1, i2;
    for (int i = 0; i < 3; i++) {
      i1 = (x.ind() + i + 1) % 3;
      i2 = (x.ind() + i + 2) % 3;
      out.push_back(Edge(x.cell(), i1, i2));
    }
    return out;
  }
  std::vector<Facet> incident_facets(Facet x) {
    std::vector<Facet> out;
    std::vector<Edge> edges = incident_edges(x);
    for (uint32_t i = 0; i < edges.size(); i++) {
      Facet_circulator cc = T.incident_facets(edges[i]._x), done(cc);
      if (cc != 0) {
	do {
	  if (Facet(cc) != x)
	    out.push_back(Facet(cc));
	} while (++cc != done);
      }
    } 
    return out;
  }
  std::vector<Cell> incident_cells(Facet x) {
    std::vector<Cell> out;
    out.push_back(x.cell());
    out.push_back(x.cell().neighbor(x.ind()));
    return out;
  }

  // Constructs incident to a cell
  std::vector<Vertex> incident_vertices(Cell x) {
    std::vector<Vertex> out;
    for (int i = 0; i < 4; i++)
      out.push_back(x.vertex(i));
    return out;
  }
  std::vector<Edge> incident_edges(Cell x) {
    std::vector<Edge> out;
    int i1, i2;
    for (i1 = 0; i1 < 4; i1++) {
      for (i2 = (i1+1); i2 < 4; i2++) {
	out.push_back(Edge(x, i1, i2));
      }
    }
    return out;
  }
  std::vector<Facet> incident_facets(Cell x) {
    std::vector<Facet> out;
    for (int i = 0; i < 4; i++)
      out.push_back(Facet(x, i));
    return out;
  }
  std::vector<Cell> incident_cells(Cell x) {
    std::vector<Cell> out;
    for (int i = 0; i < 3; i++)
      out.push_back(x.neighbor(i));
    return out;
  }

  Vertex nearest_vertex(double* pos) {
    Point p = Point(pos[0], pos[1], pos[2]);
    Vertex out = Vertex(T.nearest_vertex(p));
    return out;
  }

  void circumcenter(Cell x, double* out) {
    Point p = x._x->circumcenter();
    out[0] = p.x();
    out[1] = p.y();
    out[2] = p.z();
  }

  double dual_volume(const Vertex v) {
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

  double length(const Edge e) {
    if (is_infinite(e))
      return -1.0;
    Vertex_handle v1 = e.v1()._x;
    Vertex_handle v2 = e.v2()._x;
    Point p1 = v1->point();
    Point p2 = v2->point();
    double out = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, p2)));
    return out;
  }

  bool flip(Cell x, int i) {
    return T.flip(x._x, i);
  }
  void flip_flippable(Cell x, int i) {
    T.flip_flippable(x._x, i);
  }

  void write_to_file(const char* filename)
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      // Header
      int n = static_cast<int>(T.number_of_vertices());
      int m = static_cast<int>(T.tds().number_of_cells());
      int d = static_cast<int>(T.dimension());
      os.write((char*)&n, sizeof(int));
      os.write((char*)&m, sizeof(int));
      os.write((char*)&d, sizeof(int));
      if (n==0) {
	os.close();
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

      os.close();
    }
  }
  void read_from_file(const char* filename)
  {
    std::ifstream is(filename, std::ios::binary);
    if (!is) std::cerr << "Error cannot open file: " << filename << std::endl;
    else {

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

      is.close();
    }
  }

  void info_ordered_vertices(double* pos) {
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

  void vertex_info(Info* verts) {
    int i = 0;
    for (Finite_vertices_iterator it = T.finite_vertices_begin(); it != T.finite_vertices_end(); it++) {
      verts[i] = it->info();
      i++;
    }
  }
  
  void edge_info(Info* edges) {
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

  void outgoing_points(double *left_edge, double *right_edge, bool periodic,
                       std::vector<Info>& lx, std::vector<Info>& ly, std::vector<Info>& lz,
                       std::vector<Info>& rx, std::vector<Info>& ry, std::vector<Info>& rz,
                       std::vector<Info>& alln)
  {
    Vertex_handle v;
    Info hv;
    Point cc, p1;
    double cr;
    int i;

    for (All_cells_iterator it = T.all_cells_begin(); it != T.all_cells_end(); it++) {
      if (T.is_infinite(it) == true) {
        if (periodic == false) {
          for (i = 0; i < 4; i++) {
            v = it->vertex(i);
            if (T.is_infinite(v) == false) {
              hv = v->info();
              alln.push_back(hv);
            }
          }
        }
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

