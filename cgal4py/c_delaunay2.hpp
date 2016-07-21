#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/squared_distance_2.h>
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

template <typename Info_>
class Delaunay_with_info_2
// class Delaunay_with_info_2 : public CGAL::Delaunay_triangulation_2<K, CGAL::Triangulation_data_structure_2<CGAL::Triangulation_vertex_base_with_info_2<Info_, K>>>
{
 public:
  typedef CGAL::Delaunay_triangulation_2<K, CGAL::Triangulation_data_structure_2<CGAL::Triangulation_vertex_base_with_info_2<Info_, K>>> Delaunay;
  typedef Info_ Info;
  typedef typename Delaunay::Point                     Point;
  typedef typename Delaunay::Vertex_handle             Vertex_handle;
  typedef typename Delaunay::Face_handle               Face_handle;
  typedef typename Delaunay::Vertex_circulator         Vertex_circulator;
  typedef typename Delaunay::Face_circulator           Face_circulator;
  typedef typename Delaunay::Vertex_iterator           Vertex_iterator;
  typedef typename Delaunay::Face_iterator             Face_iterator;
  typedef typename Delaunay::Edge_iterator             Edge_iterator;
  typedef typename Delaunay::All_vertices_iterator     All_vertices_iterator;
  typedef typename Delaunay::All_faces_iterator        All_faces_iterator;
  typedef typename Delaunay::All_edges_iterator        All_edges_iterator;
  typedef typename Delaunay::Finite_vertices_iterator  Finite_vertices_iterator;
  typedef typename Delaunay::Finite_edges_iterator     Finite_edges_iterator;
  typedef typename Delaunay::Triangle                  Triangle;
  typedef typename CGAL::Unique_hash_map<Vertex_handle,int>  Vertex_hash;
  typedef typename CGAL::Unique_hash_map<Face_handle,int>    Face_hash;
  Delaunay T;
  Delaunay_with_info_2() {};
  Delaunay_with_info_2(double *pts, Info *val, uint32_t n) { insert(pts, val, n); }
  uint32_t num_finite_verts() { return static_cast<uint32_t>(T.number_of_vertices()); }
  uint32_t num_finite_cells() { return static_cast<uint32_t>(T.number_of_faces()); }
  uint32_t num_infinite_verts() { return static_cast<uint32_t>(1); }
  uint32_t num_infinite_cells() {
    Face_circulator fc = T.incident_faces(T.infinite_vertex()), done(fc);
    if (fc == 0)
      return 0;
    int count = 0;
    do {
      count++;
    } while (++fc != done);
    return count;
  }
  uint32_t num_verts() { return (num_finite_verts() + num_infinite_verts()); }
  uint32_t num_cells() { return (num_finite_cells() + num_infinite_cells()); }

  class All_verts_iter {
  public:
    All_vertices_iterator _x = All_vertices_iterator();
    All_verts_iter() { _x = All_vertices_iterator(); }
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
  All_verts_iter all_verts_begin() {
    return All_verts_iter(T.all_vertices_begin());
  }
  All_verts_iter all_verts_end() {
    return All_verts_iter(T.all_vertices_end()); 
  }

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
    }
    std::vector<double> point() {
      std::vector<double> out;
      Point p = _x->point();
      out.push_back(p.x());
      out.push_back(p.y());
      return out;
    }
    Info info() {
      return _x->info();
    }
  };

  class All_cells_iter {
  public:
    All_faces_iterator _x = All_faces_iterator();
    All_cells_iter() { _x = All_faces_iterator(); }
    All_cells_iter(All_faces_iterator x) { _x = x; }
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
  All_cells_iter all_cells_begin() { return All_cells_iter(T.all_faces_begin()); }
  All_cells_iter all_cells_end() { return All_cells_iter(T.all_faces_end()); }

  class Cell_circ {
  public:
    Face_circulator _x = Face_circulator();
    Face_circulator _done = _x;
    Cell_circ() {
      _x = Face_circulator();
      _done = _x;
    }
    Cell_circ(Face_circulator x) { _x = x; _done = x; }
    Cell_circ& operator*() { return *this; }
    Cell_circ& operator++() {
      _x++;
      return *this;
    }
    Cell_circ& operator--() {
      _x--;
      return *this;
    }
    bool operator==(Cell_circ other) { return (_x == other._x); }
    bool operator!=(Cell_circ other) { return (_x != other._x); }
    bool is_done() { return (_x == _done); }
  };

  class Cell {
  public:
    Face_handle _x = Face_handle();
    Cell() { _x = Face_handle(); }
    Cell(Face_handle x) { _x = x; }
    Cell(All_cells_iter x) { _x = static_cast<Face_handle>(x._x); }
    Cell(Cell_circ x) { _x = static_cast<Face_handle>(x._x); }
    bool operator==(Cell other) { return (_x == other._x); }
    bool operator!=(Cell other) { return (_x != other._x); }
  };

  bool is_infinite(Vertex x) { return T.is_infinite(x._x); }
  bool is_infinite(Cell x) { return T.is_infinite(x._x); }
  bool is_infinite(All_verts_iter x) { return T.is_infinite(x._x); }
  bool is_infinite(All_cells_iter x) { return T.is_infinite(x._x); }

  std::vector<Cell> incident_cells(Vertex x) {
    std::vector<Cell> out;
    Face_circulator fc = T.incident_faces(x._x), done(fc);
    if (fc == 0)
      return out;
    do {
      out.push_back(Cell(static_cast<Face_handle>(fc)));
    } while (++fc != done);
    return out;
  }
  
  std::vector<Vertex> incident_vertices(Vertex x) {
    std::vector<Vertex> out;
    Vertex_circulator vc = T.incident_vertices(x._x), done(vc);
    if (vc == 0)
      return out;
    do {
      out.push_back(Vertex(static_cast<Vertex_handle>(vc)));
    } while (++vc != done);
    return out;
  }

  Vertex nearest_vertex(double* pos) {
    Point p = Point(pos[0], pos[1]);
    Vertex out = Vertex(T.nearest_vertex(p));
    return out;
  }

  void circumcenter(Cell x, double* out) {
    Point p = T.circumcenter(x._x);
    out[0] = p.x();
    out[1] = p.y();
  }

  double dual_area(const Vertex v) {

    Face_circulator fstart = T.incident_faces(v._x);
    Face_circulator fcit = fstart;
    std::vector<Point> pts;


    pts.push_back(T.circumcenter(fstart));
    fcit++;
    for ( ; fcit != fstart; fcit++) {
      if (T.is_infinite(fcit))
	return -1.0;
      Point dual = T.circumcenter(fcit);
      pts.push_back(dual);
    }
    pts.push_back(T.circumcenter(fstart));

    double vol = 0.0;
    Point orig = v._x->point();
    for (uint32_t i=0 ; i<pts.size()-1 ; i++) {
      vol = vol + Triangle(orig,pts[i],pts[i+1]).area();
    }

    return vol;
  }

  void write_to_file(const char* filename)
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      // Header
      int n = static_cast<int>(T.number_of_vertices());
      int m = static_cast<int>(T.tds().number_of_full_dim_faces());
      int d = static_cast<int>(T.dimension());
      os.write((char*)&n, sizeof(int));
      os.write((char*)&m, sizeof(int));
      os.write((char*)&d, sizeof(int));
      if (n==0) {
	os.close();
	return;
      }

      Vertex_hash V;
      Face_hash F;
      
      // first (infinite) vertex 
      Info info;
      double x, y;
      int inum = 0;
      Vertex_handle v = T.infinite_vertex();
      if ( v != Vertex_handle()) {
	V[v] = inum++;
      }
      
      // other vertices
      for( All_vertices_iterator vit = T.tds().vertices_begin(); vit != T.tds().vertices_end() ; ++vit) {
	if ( v != vit ) {
	  V[vit] = inum++;
	  info = static_cast<Info>(vit->info());
	  x = static_cast<double>(vit->point().x());
	  y = static_cast<double>(vit->point().y());
	  os.write((char*)&info, sizeof(Info));
	  os.write((char*)&x, sizeof(double));
	  os.write((char*)&y, sizeof(double));
	}
      }
      
      // vertices of the faces
      inum = 0;
      int dim = (d == -1 ? 1 :  d + 1);
      int index;
      int nvert = 0;
      for (All_faces_iterator ib = T.tds().face_iterator_base_begin();
	   ib != T.tds().face_iterator_base_end(); ++ib) {
	F[ib] = inum++;
	for (int j = 0; j < dim ; ++j) {
	  index = V[ib->vertex(j)];
	  os.write((char*)&index, sizeof(int));
	  nvert++;
	}
      }
  
      // neighbor pointers of the faces
      for (All_faces_iterator it = T.tds().face_iterator_base_begin();
	   it != T.tds().face_iterator_base_end(); ++it) {
	for (int j = 0; j < d+1; ++j){
	  index = F[it->neighbor(j)];
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
	T.set_infinite_vertex(Vertex_handle());
	return;
      }

      T.tds().set_dimension(d);

      All_faces_iterator to_delete = T.tds().face_iterator_base_begin();

      std::vector<Vertex_handle> V(n+1);
      std::vector<Face_handle> F(m);

      // infinite vertex
      int i = 0;
      V[0] = T.infinite_vertex();
      ++i;

      // read vertices
      Info info;
      double x, y;
      for( ; i < (n+1); ++i) {
	V[i] = T.tds().create_vertex();
	is.read((char*)&info, sizeof(Info));
	is.read((char*)&x, sizeof(double));
	is.read((char*)&y, sizeof(double));
	(*(V[i])).point() = Point(x,y);
	(*(V[i])).info() = info;
      }

      // Creation of the faces
      int index;
      int dim = (d == -1 ? 1 : d + 1);
      {
	int nvert = 0;
	for(i = 0; i < m; ++i) {
	  F[i] = T.tds().create_face() ;
	  for(int j = 0; j < dim ; ++j){
	    is.read((char*)&index, sizeof(int));
	    F[i]->set_vertex(j, V[index]);
	    // The face pointer of vertices is set too often,
	    // but otherwise we had to use a further map 
	    V[index]->set_face(F[i]);
	    nvert++;
	  }
	}
      }

      // Setting the neighbor pointers
      {
	for(i = 0; i < m; ++i) {
	  for(int j = 0; j < d+1; ++j){
	    is.read((char*)&index, sizeof(int));
	    F[i]->set_neighbor(j, F[index]);
	  }
	}
      }

      // Remove flat face
      T.tds().delete_face(to_delete);

      T.set_infinite_vertex(V[0]);
      is.close();
    }
  }

  void insert(double *pts, Info *val, uint32_t n)
  {
    uint32_t i, j;
    std::vector< std::pair<Point,Info> > points;
    for (i = 0; i < n; i++) {
      j = 2*i;
      points.push_back( std::make_pair( Point(pts[j],pts[j+1]), val[i]) );
    }
    T.insert( points.begin(),points.end() );
  }
  void edge_info(std::vector< std::pair<Info,Info> >& edges)
  {
    Info i1, i2;
    for (Finite_edges_iterator it = T.finite_edges_begin(); it != T.finite_edges_end(); it++) {
      i1 = it->first->vertex(T.cw(it->second))->info();
      i2 = it->first->vertex(T.ccw(it->second))->info();
      edges.push_back( std::make_pair( i1, i2 ) );
    }
  }
  void outgoing_points(double *left_edge, double *right_edge, bool periodic,
                       std::vector<Info>& lx, std::vector<Info>& ly,
                       std::vector<Info>& rx, std::vector<Info>& ry,
                       std::vector<Info>& alln)
  {
    Vertex_handle v;
    Info hv;
    Point cc, p1;
    double cr;
    int i;

    for (All_faces_iterator it = T.all_faces_begin(); it != T.all_faces_end(); it++) {
      if (T.is_infinite(it) == true) {
        if (periodic == false) {
          for (i = 0; i < 3; i++) {
            v = it->vertex(i);
            if (T.is_infinite(v) == false) {
              hv = v->info();
              alln.push_back(hv);
            }
          }
        }
      } else {
        p1 = it->vertex(0)->point();
        cc = T.circumcenter(it);
        cr = std::sqrt(static_cast<double>(CGAL::squared_distance(p1, cc)));
        if ((cc.x() + cr) > right_edge[0])
          for (i = 0; i < 3; i++) rx.push_back((it->vertex(i))->info());
        if ((cc.y() + cr) > right_edge[1])
          for (i = 0; i < 3; i++) ry.push_back((it->vertex(i))->info());
        if ((cc.x() - cr) < left_edge[0])
          for (i = 0; i < 3; i++) lx.push_back((it->vertex(i))->info());
        if ((cc.y() - cr) < left_edge[1])
          for (i = 0; i < 3; i++) ly.push_back((it->vertex(i))->info());
      }
    }
    std::sort( alln.begin(), alln.end() );
    std::sort( lx.begin(), lx.end() );
    std::sort( ly.begin(), ly.end() );
    std::sort( rx.begin(), rx.end() );
    std::sort( ry.begin(), ry.end() );
    alln.erase( std::unique( alln.begin(), alln.end() ), alln.end() );
    lx.erase( std::unique( lx.begin(), lx.end() ), lx.end() );
    ly.erase( std::unique( ly.begin(), ly.end() ), ly.end() );
    rx.erase( std::unique( rx.begin(), rx.end() ), rx.end() );
    ry.erase( std::unique( ry.begin(), ry.end() ), ry.end() );
    
  }
};

