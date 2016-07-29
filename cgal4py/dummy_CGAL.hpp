#include <vector>
#include <utility>
#include <memory>
#include <iterator>
#include <list>

namespace CGAL 
{
enum  Sign
  {   
    NEGATIVE = -1, ZERO = 0, POSITIVE = 1,
    RIGHT_TURN = -1, LEFT_TURN = 1,
    CLOCKWISE = -1, COUNTERCLOCKWISE = 1,
    COLLINEAR = 0, COPLANAR = 0, DEGENERATE = 0,
    ON_NEGATIVE_SIDE = -1, ON_ORIENTED_BOUNDARY = 0, ON_POSITIVE_SIDE = 1,
    SMALLER = -1, EQUAL = 0, LARGER = 1
  };
  typedef Sign Orientation;
  typedef Sign Oriented_side;
  typedef Sign Comparison_result;
  enum  Bounded_side
    {
      ON_UNBOUNDED_SIDE = -1,
      ON_BOUNDARY,
      ON_BOUNDED_SIDE
    };

  template <class Key_, class Data_>
  class Unique_hash_map {
  private:
    Data_ _data;
  public:
    typedef Key_                                     Key;
    typedef Data_                                    Data;
    Unique_hash_map() 
      : _data(0)
    {};
    const Data& operator[]( const Key& key) const { return _data; }
    Data& operator[]( const Key& key) { return _data; }
  };

  template <class T1, class T2, class T3>
  class Triple {
    typedef Triple<T1, T2, T3> Self;
  public:
    typedef T1 first_type;
    typedef T2 second_type;
    typedef T3 third_type;
    T1 first;
    T2 second;
    T3 third;
    Triple() {};
    Triple(const T1& a, const T2& b, const T3& c) {};
    template <class U, class V, class W>
    Triple(const U& a, const V& b, const W& c) {};
    template <class U, class V, class W>
    Triple& operator=(const Triple<U, V, W> &t) { return *this; }
    bool operator==(const Self other) const { return false; }
    bool operator!=(const Self other) const { return false; }
  };

  template <class K>
  double squared_distance(K& p1, K& p2) {
    double x = 0.0;
    return x;
  }

  class Exact_predicates_inexact_constructions_kernel {
  public:
    class Point {
    public:
      Point() {};
      Point(double x, double y) {};
      Point(double x, double y, double z) {};
      double x() { return 0.0; }
      double y() { return 0.0; }
      double z() { return 0.0; }
    };

    class Segment {
    public:
      Segment() {};
      Segment(Point p1, Point p2) {};
    };

    class Triangle {
    public:
      Triangle() {};
      Triangle(Point p1, Point p2, Point p3) {};
      double area() { return 0.0; }
    };

    class Tetrahedron {
    public:
      Tetrahedron() {};
      Tetrahedron(Point p1, Point p2, Point p3, Point p4) {};
      double volume() { return 0.0; }
    };
  };

  template < class H, bool Const >
  class Iterator;

  template < class T >
  class Container {
  public:
    typedef Container<T>         Self;
    typedef T                    value_type;
    typedef Iterator<Self,false> iterator;
    typedef Iterator<Self,true>  const_iterator;
    
    friend class Iterator<Self,false>;
    friend class Iterator<Self,true>;

    Container() {};
    Container & operator=(const Container &c) {};
    
    iterator begin() { return iterator(); }
    iterator end() { return iterator(); }

    const_iterator begin() const { return const_iterator(); }
    const_iterator end() const { return const_iterator(); }
    
  };
  
  template < class T >
  bool operator==(const Container<T> &lhs, const Container<T> &rhs) { return false; }
  template < class T >
  bool operator!=(const Container<T> &lhs, const Container<T> &rhs) { return false; }
  
  
  template < class H, bool Const >
  class Iterator {
  public:
    typedef Iterator<H, Const> Self;
    typedef typename H::iterator iterator;
    typedef typename H::value_type value_type;
    typedef typename std::conditional<Const, 
				      const value_type*, 
				      value_type*>::type pointer;
    typedef typename std::conditional<Const, 
				      const value_type&, 
				      value_type&>::type reference;
  private:
    pointer p;
  public:    
    Iterator() { p = NULL; }
    // Iterator(const iterator &it) { p = &(*it); }
    Self & operator= (const iterator &it) { return *this; }
    Self & operator++() { return *this; }
    Self & operator--() { return *this; }
    Self operator++(int) { Self tmp(*this); ++(*this); return tmp; }
    Self operator--(int) { Self tmp(*this); --(*this); return tmp; }
    reference operator*() const { return *(p); }
    pointer   operator->() const { return p; }
  };
  
  template < class H, bool Const >
  bool operator==(const Iterator<H,Const> &lhs, const Iterator<H,Const> &rhs) { return false; }
  template < class H, bool Const >
  bool operator!=(const Iterator<H,Const> &lhs, const Iterator<H,Const> &rhs) { return false; }

  template < class I >
  class Filter_iterator {
  public:
    typedef Filter_iterator<I> Self;
    typedef I                                Iterator;
    typedef typename I::reference            reference;
    typedef typename I::pointer              pointer;
    typedef typename I::value_type           value_type;
  private:
    Iterator e_;
    Iterator c_;
  public:    
    Filter_iterator()
      : e_(0), c_(0) 
    {}
    Filter_iterator(Iterator e) 
      : e_(e), c_(e)
    {}
    Self & operator++() { return *this; }
    Self & operator--() { return *this; }
    Self operator++(int) { Self tmp(*this); ++(*this); return tmp; }
    Self operator--(int) { Self tmp(*this); --(*this); return tmp; }
    reference operator*() const { return *c_;  }
    pointer operator->() const  { return &*c_; }
    Iterator base() const { return c_; }
    operator Iterator() { return e_; }
  };

  template < class I >
  bool operator==(const Filter_iterator<I> &lhs, const Filter_iterator<I> &rhs) { return false; }
  template < class I >
  bool operator!=(const Filter_iterator<I> &lhs, const Filter_iterator<I> &rhs) { return false; }

  template < class Tds, typename Info_ >
  class Vertex_base_2 {
    typedef Info_                                      Info;
    typedef typename Tds::Face_handle                  Face_handle;
    typedef typename Tds::Vertex_handle                Vertex_handle;
  // public:
    typedef typename Exact_predicates_inexact_constructions_kernel::Point Point;
  private:
    Point _p;
    Info_ _info;
  public:
    Face_handle face() const { return Face_handle(); }
    void set_face(Face_handle f) {};
    bool is_valid(bool /*verbose*/=false, int /*level*/= 0) const { return false; }
    void set_point(const Point & p) {};
    const Point&  point() const { return _p; }
    Point&        point() { return _p; }
    const Info& info() const { return _info; }
    Info&       info() { return _info; }
  };

  template < class Tds, typename Info_ >
  class Vertex_base_3 {
    typedef Info_                                      Info;
    typedef typename Tds::Cell_handle                  Cell_handle;
    typedef typename Tds::Vertex_handle                Vertex_handle;
  // public:
    typedef typename Exact_predicates_inexact_constructions_kernel::Point Point;
  private:
    Point _p;
    Info_ _info;
  public:
    Cell_handle cell() const { return Cell_handle(); }
    void set_cell(Cell_handle f) {};
    bool is_valid(bool /*verbose*/=false, int /*level*/= 0) const { return false; }
    void set_point(const Point & p) {};
    const Point&  point() const { return _p; }
    Point&        point() { return _p; }
    const Info& info() const { return _info; }
    Info&       info() { return _info; }
  };

  template < class Tds >
  class Face_base {
    typedef typename Tds::Face_handle                  Face_handle;
    typedef typename Tds::Vertex_handle                Vertex_handle;
    typedef typename Exact_predicates_inexact_constructions_kernel::Point Point;
  public:
    Face_base() {};
    Face_base(Vertex_handle v0,
	      Vertex_handle v1,
	      Vertex_handle v2) {};
    Face_base(Vertex_handle v0,
	      Vertex_handle v1,
	      Vertex_handle v2,
	      Face_handle n0,
	      Face_handle n1,
	      Face_handle n2) {};
    Vertex_handle vertex(int i) const { return Vertex_handle(); }
    bool has_vertex(Vertex_handle v) const { return false; }
    bool has_vertex(Vertex_handle v, int& i) const { return false; }
    int index(Vertex_handle v) const { return 0; }
    Face_handle neighbor(int i) const { return Face_handle(); }
    bool has_neighbor(Face_handle n) const { return false; }
    bool has_neighbor(Face_handle n, int& i) const { return false; }
    int index(Face_handle n) const { return 0; }
    void set_vertex(int i, Vertex_handle v) {};
    void set_vertices() {};
    void set_vertices(Vertex_handle v0, Vertex_handle v1, Vertex_handle v2) {};
    void set_neighbor(int i, Face_handle n) {};
    void set_neighbors() {};
    void set_neighbors(Face_handle n0, Face_handle n1, Face_handle n2) {};
    void reorient() {};
    void ccw_permute() {};
    void cw_permute() {};
    int dimension() const { return 0; }
    bool is_valid(bool /*verbose*/=false, int /*level*/= 0) const { return false; }
  };

  template < class Tds >
  class Cell_base {
    typedef typename Tds::Cell_handle                  Cell_handle;
    typedef typename Tds::Vertex_handle                Vertex_handle;
    typedef typename Exact_predicates_inexact_constructions_kernel::Point Point;
  public:
    Cell_base() {};
    Cell_base(Vertex_handle v0,
	      Vertex_handle v1,
	      Vertex_handle v2,
	      Vertex_handle v3) {};
    Cell_base(Vertex_handle v0,
	      Vertex_handle v1,
	      Vertex_handle v2,
	      Vertex_handle v3,
	      Cell_handle n0,
	      Cell_handle n1,
	      Cell_handle n2,
	      Cell_handle n3) {};
    Vertex_handle vertex(int i) const { return Vertex_handle(); }
    bool has_vertex(Vertex_handle v) const { return false; }
    bool has_vertex(Vertex_handle v, int& i) const { return false; }
    int index(Vertex_handle v) const { return 0; }
    Cell_handle neighbor(int i) const { return Cell_handle(); }
    bool has_neighbor(Cell_handle n) const { return false; }
    bool has_neighbor(Cell_handle n, int& i) const { return false; }
    int index(Cell_handle n) const { return 0; }
    void set_vertex(int i, Vertex_handle v) {};
    void set_vertices() {};
    void set_vertices(Vertex_handle v0, Vertex_handle v1, Vertex_handle v2, Vertex_handle v3) {};
    void set_neighbor(int i, Cell_handle n) {};
    void set_neighbors() {};
    void set_neighbors(Cell_handle n0, Cell_handle n1, Cell_handle n2, Cell_handle n3) {};
    void reorient() {};
    void ccw_permute() {};
    void cw_permute() {};
    int dimension() const { return 0; }
    bool is_valid(bool /*verbose*/=false, int /*level*/= 0) const { return false; }
    Point circumcenter() const { return Point(); }
  };

  // Things specific to 2D case
  template <class Tds>
  class Triangulation_ds_edge_iterator_2
  {
  public:
    typedef typename Tds::Edge           Edge;
    typedef typename Tds::Face_iterator  Face_iterator;
    typedef typename Tds::Face_handle    Face_handle;
    typedef Triangulation_ds_edge_iterator_2<Tds> Edge_iterator;
  private:
    const Tds* _tds;
    Face_iterator pos;
    mutable Edge edge;
  public:
    typedef Edge            value_type;
    typedef Edge*           pointer;
    typedef Edge&           reference;
    Triangulation_ds_edge_iterator_2() {};
    bool operator==(const Edge_iterator& fi) const { return false; }
    bool operator!=(const Edge_iterator& fi) const { return false; }
    Edge_iterator& operator++() { return *this; }
    Edge_iterator& operator--() { return *this; }
    Edge_iterator operator++(int) {  
      Edge_iterator tmp(*this);
      ++(*this);
      return tmp;
    }
    Edge_iterator operator--(int) {
      Edge_iterator tmp(*this);
      --(*this);
      return tmp;
    }
    Edge* operator->() const {
      edge.first = pos;
      return &edge;
    }
    Edge& operator*() const {
      edge.first = pos;
      return edge;
    }
  };

  template < class Tds>
  class Triangulation_ds_face_circulator_2 
  {
  public:
    typedef Triangulation_ds_face_circulator_2<Tds> Face_circulator;
    typedef typename Tds::Face                      Face;
    typedef typename Tds::Vertex                    Vertex;
    typedef typename Tds::Face_handle               Face_handle;
    typedef typename Tds::Vertex_handle             Vertex_handle;
  private:
    Vertex_handle _v;
    Face_handle    pos;
  public:
    Triangulation_ds_face_circulator_2() 
      : _v(), pos() 
    {}
    Triangulation_ds_face_circulator_2(Vertex_handle v,
				       Face_handle f = Face_handle()) {
      _v = v;
      pos = f;
    }
    Face_circulator& operator=(const Face_circulator& other) { return *this; }
    Face_circulator& operator++() { return *this; }
    Face_circulator operator++(int) {
      Face_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    Face_circulator& operator--() { return *this; }
    Face_circulator operator--(int) {
      Face_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    operator Face_handle() const { return pos; }
    bool operator==(const Face_circulator &fc) const { return false; }
    bool operator!=(const Face_circulator &fc) const { return false; }
    bool operator==(const int i) const { return false; }
    bool operator==(const Face_handle &fh) const { return pos == fh; }
    bool operator!=(const Face_handle &fh) const { return pos != fh; }
    Face& operator*() const { return *pos; }
    Face* operator->() const { return &*pos; }
  };
  template < class Tds_ >
  bool operator==(typename Tds_::Face_handle fh,
		  Triangulation_ds_face_circulator_2<Tds_> fc) { return (fc==fh); }
  template < class Tds_ >
  bool operator!=(typename Tds_::Face_handle fh,
		  Triangulation_ds_face_circulator_2<Tds_> fc) { return (fc!=fh); }

  template < class Tds >
  class Triangulation_ds_vertex_circulator_2 {
  public:
    typedef Triangulation_ds_vertex_circulator_2<Tds> Vertex_circulator;
    typedef typename Tds::Face                      Face;
    typedef typename Tds::Vertex                    Vertex;
    typedef typename Tds::Face_handle               Face_handle;
    typedef typename Tds::Vertex_handle             Vertex_handle;
  private:
    Vertex_handle _v;
    Face_handle   pos;
    int _ri;
  public:
    Triangulation_ds_vertex_circulator_2()
      :  _v(), pos()
    {}
    Triangulation_ds_vertex_circulator_2(Vertex_handle v,
					 Face_handle f = Face_handle()) 
      : _v(v), pos(pos)
    {}

    Vertex_circulator& operator++() { return *this; }
    Vertex_circulator  operator++(int) {
      Vertex_circulator tmp(*this);
      ++(*this);
      return tmp;
    }
    Vertex_circulator& operator--() { return *this; }
    Vertex_circulator  operator--(int) {
      Vertex_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    bool operator==(const int i) const { return false; }
    bool operator==(const Vertex_circulator &vc) const { return false; }
    bool operator!=(const Vertex_circulator &vc) const { return false; }
    bool operator==(const Vertex_handle &vh) const { return false; }
    bool operator!=(const Vertex_handle &vh) const{ return false; }
    Vertex& operator*() const { return *(pos->vertex(_ri)); }
    Vertex* operator->() const { return &*(pos->vertex(_ri)); }
    operator Vertex_handle() const { return _v; }
  };

  template < class Tds_ >
  bool operator==(typename Tds_::Vertex_handle vh,
		  Triangulation_ds_vertex_circulator_2<Tds_> vc) { return (vc==vh); }

  template < class Tds_ >
  bool operator!=(typename Tds_::Vertex_handle vh,
		  Triangulation_ds_vertex_circulator_2<Tds_> vc) { return !(vc==vh); }

  template < class Tds >
  class Triangulation_ds_edge_circulator_2 {
  public:
    typedef Triangulation_ds_edge_circulator_2<Tds>  Edge_circulator;
    typedef typename Tds::Face                       Face;
    typedef typename Tds::Vertex                     Vertex;
    typedef typename Tds::Edge                       Edge;
    typedef typename Tds::Face_handle                Face_handle;
    typedef typename Tds::Vertex_handle              Vertex_handle;

  private:
    int _ri;
    Vertex_handle _v;
    Face_handle  pos;
    mutable Edge edge;

  public:
    Triangulation_ds_edge_circulator_2()
      : _ri(0), _v(), pos()
    {}
    Triangulation_ds_edge_circulator_2( Vertex_handle v,
					Face_handle f = Face_handle()) 
      : _ri(0), _v(v), pos(f)
    {}

    Edge_circulator& operator++() { return *this; }
    Edge_circulator operator++(int) {
      Edge_circulator tmp(*this);
      ++(*this);
      return tmp;
    }
    Edge_circulator& operator--() { return *this; }
    Edge_circulator operator--(int) {
      Edge_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    bool operator==(const int i) const { return false; }
    bool operator==(const Edge_circulator &vc) const { return false; }
    bool operator!=(const Edge_circulator &vc) const { return false; }
    bool is_empty() const { return false; }
    Edge*  operator->() const {
      edge.first=pos;
      edge.second= _ri;
      return &edge;
    }
    Edge& operator*() const {
      edge.first=pos;
      edge.second= _ri;
      return edge;
    }
  };

  template < typename Info_, typename GT>
  class Triangulation_vertex_base_with_info_2 {
    Info_ _info;
  public:
    typedef Info_ Info;
  };

  template < class Vb >
  class Triangulation_data_structure_2 {
  public:
    typedef Triangulation_data_structure_2<Vb>         Tds;
    typedef typename Vb::Info                          Info;
    typedef Vertex_base_2<Tds,Info>                    Vertex;
    typedef Face_base<Tds>                             Face;

    friend class Triangulation_ds_edge_iterator_2<Tds>;
    friend class Triangulation_ds_face_circulator_2<Tds>;
    friend class Triangulation_ds_edge_circulator_2<Tds>;
    friend class Triangulation_ds_vertex_circulator_2<Tds>;

    typedef Triangulation_ds_edge_iterator_2<Tds>      Edge_iterator;
    typedef Triangulation_ds_face_circulator_2<Tds>    Face_circulator;
    typedef Triangulation_ds_vertex_circulator_2<Tds>  Vertex_circulator;
    typedef Triangulation_ds_edge_circulator_2<Tds>    Edge_circulator;
    
    typedef Container<Vertex>                          Vertex_range;
    typedef Container<Face>                            Face_range;
    typedef typename Face_range::iterator              Face_iterator;
    typedef typename Vertex_range::iterator            Vertex_iterator;
    typedef Vertex_iterator                            Vertex_handle;
    typedef Face_iterator                              Face_handle;
    typedef std::pair<Face_handle, int>                Edge;
    typedef std::list<Edge>                            List_edges;

  protected:
    Face_range   _faces;
    Vertex_range _vertices;

  public:
    Triangulation_data_structure_2() {};
    Face_range& faces() { return _faces; }
    Face_range& faces() const { return const_cast<Tds*>(this)->_faces; }
    Vertex_range& vertices() { return _vertices; }
    Vertex_range& vertices() const { return const_cast<Tds*>(this)->_vertices; }

    int dimension() const { return 0; }
    int number_of_vertices() const { return 0; }
    int number_of_faces() const { return 0; }
    int number_of_edges() const { return 0; }
    int number_of_full_dim_faces() const { return 0; }
    
    bool is_vertex(Vertex_handle v) const { return false; }
    bool is_edge(Face_handle fh, int i) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb,
		 Face_handle& fr,  int& i) const { return false; }
    bool is_face(Face_handle fh) const { return false; }
    bool is_face(Vertex_handle v1,
		 Vertex_handle v2,
		 Vertex_handle v3) const { return false; }
    bool is_face(Vertex_handle v1,
		 Vertex_handle v2,
		 Vertex_handle v3,
		 Face_handle& fr) const { return false; }

    Face_iterator faces_begin() const { return faces_end(); }
    Face_iterator faces_end() const { return faces().end(); }
    Vertex_iterator vertices_begin() const { return vertices_end(); }
    Vertex_iterator vertices_end() const { return vertices().end(); }
    Edge_iterator edges_begin() const { return Edge_iterator(); }
    Edge_iterator edges_end() const { return Edge_iterator(); }
    Face_iterator face_iterator_base_begin() const { return faces().end(); }
    Face_iterator face_iterator_base_end() const { return faces().end(); }

    Face_circulator incident_faces(Vertex_handle v,
				   Face_handle f =  Face_handle()) const { return Face_circulator(); }
    Vertex_circulator incident_vertices(Vertex_handle v,
					Face_handle f = Face_handle()) const { return Vertex_circulator(); }
    Edge_circulator incident_edges(Vertex_handle v,
				   Face_handle f = Face_handle()) const { return Edge_circulator(); }

    Vertex_handle mirror_vertex(Face_handle f, int i) const { return Vertex_handle(); }
    int mirror_index(Face_handle f, int i) const { return 0; }
    Edge mirror_edge(const Edge e) const { return Edge(); }

    void flip(Face_handle f, int i) {};
    Vertex_handle insert_first() { return Vertex_handle(); }
    Vertex_handle insert_second() { return Vertex_handle(); }
    Vertex_handle insert_in_face(Face_handle f) { return Vertex_handle(); }
    Vertex_handle insert_in_edge(Face_handle f, int i) { return Vertex_handle(); }
    Vertex_handle insert_dim_up(Vertex_handle w = Vertex_handle(),
				bool orient=true) { return Vertex_handle(); }
    void remove_degree_3(Vertex_handle v, Face_handle f = Face_handle()) {};
    void remove_1D(Vertex_handle v) {};
    void remove_second(Vertex_handle v) {};
    void remove_first(Vertex_handle v) {};
    void remove_dim_down(Vertex_handle v) {};
    void dim_down(Face_handle f, int i) {};
    Vertex_handle star_hole(List_edges& hole) { return Vertex_handle(); }
    void star_hole(Vertex_handle v, List_edges& hole) {};
    void make_hole(Vertex_handle v, List_edges& hole) {};
    
    Vertex_handle create_vertex(const Vertex &v = Vertex()) { return Vertex_handle(); }
    Vertex_handle create_vertex(Vertex_handle v) { return Vertex_handle(); }
    Face_handle create_face(const Face& f = Face()) { return Face_handle(); }
    Face_handle create_face(Face_handle f) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2,
			    Face_handle f3, int i3) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1, Vertex_handle v) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3,
			    Face_handle f1,
			    Face_handle f2,
			    Face_handle f3) { return Face_handle(); }
    void set_adjacency(Face_handle f0, int i0, Face_handle f1, int i1) const {};
    void delete_face(Face_handle) {};
    void delete_vertex(Vertex_handle) {};

    void clear() {};
    void set_dimension (int n) {};

    template < class EdgeIt >
    Vertex_handle star_hole(EdgeIt edge_begin, EdgeIt edge_end) { return Vertex_handle(); }
    template < class EdgeIt >
    void star_hole(Vertex_handle v, EdgeIt edge_begin, EdgeIt edge_end) {};
    template < class EdgeIt, class FaceIt >
    Vertex_handle star_hole(EdgeIt edge_begin,
			    EdgeIt edge_end,
			    FaceIt face_begin,
			    FaceIt face_end) { return Vertex_handle(); }
    template < class EdgeIt, class FaceIt >
    void  star_hole(Vertex_handle newv,
		    EdgeIt edge_begin,
		    EdgeIt edge_end,
		    FaceIt face_begin,
		    FaceIt face_end) {};
    

  };

  template < class Gt, class Tds >
  class Delaunay_triangulation_2 {
  public:
    typedef typename Tds::Vertex                 Vertex;
    typedef typename Tds::Face                   Face;
    typedef typename Tds::Edge                   Edge;
    typedef typename Tds::Vertex_handle          Vertex_handle;
    typedef typename Tds::Face_handle            Face_handle;

    typedef typename Tds::Face_circulator        Face_circulator;
    typedef typename Tds::Vertex_circulator      Vertex_circulator;
    typedef typename Tds::Edge_circulator        Edge_circulator;

    typedef typename Tds::Face_iterator          All_faces_iterator;
    typedef typename Tds::Edge_iterator          All_edges_iterator;
    typedef typename Tds::Vertex_iterator        All_vertices_iterator;

    typedef Filter_iterator<All_faces_iterator>  Finite_faces_iterator;
    typedef Filter_iterator<All_edges_iterator>  Finite_edges_iterator;
    typedef Filter_iterator<All_vertices_iterator>  Finite_vertices_iterator;

    typedef Finite_faces_iterator                Face_iterator;
    typedef Finite_edges_iterator                Edge_iterator;
    typedef Finite_vertices_iterator             Vertex_iterator;

    enum Locate_type {VERTEX=0,
		      EDGE,
		      FACE,
		      OUTSIDE_CONVEX_HULL,
		      OUTSIDE_AFFINE_HULL};

    typedef typename Tds::Info Info;
    typedef typename Gt::Point Point;
    typedef typename Gt::Triangle Triangle;
    typedef typename Gt::Segment Segment;

  protected:
    Gt  _gt;
    Tds _tds;
    Vertex_handle _infinite_vertex;

  public:
    Delaunay_triangulation_2() {};
    void clear() {};
    const Tds & tds() const { return _tds;}
    Tds & tds() { return _tds;}
    int dimension() const { return 0; }
    int number_of_vertices() const { return 0; }
    int number_of_faces() const { return 0; }
    Vertex_handle infinite_vertex() const { return _infinite_vertex; }
    Vertex_handle finite_vertex() const { return Vertex_handle(); }
    Face_handle infinite_face() const { return Face_handle(); }
    void set_infinite_vertex(const Vertex_handle& v) {};

    bool is_valid() const { return false; }

    bool is_infinite(Face_handle f) const { return false; }
    bool is_infinite(Vertex_handle v) const { return false; }
    bool is_infinite(Face_handle f, int i) const { return false; }
    bool is_infinite(const Edge& e) const { return false; }
    bool is_infinite(const Edge_circulator& ec) const { return false; }
    bool is_infinite(const All_edges_iterator& ei) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb) const { return false; }
    bool is_edge(Vertex_handle va, Vertex_handle vb, Face_handle& fr,
		 int & i) const { return false; }
    bool includes_edge(Vertex_handle va, Vertex_handle vb,
		       Vertex_handle& vbb,
		       Face_handle& fr, int & i) const { return false; }
    bool is_face(Vertex_handle v1, Vertex_handle v2, Vertex_handle v3) const { return false; }
    bool is_face(Vertex_handle v1, Vertex_handle v2, Vertex_handle v3,
		 Face_handle &fr) const { return false; }

    Triangle triangle(Face_handle f) const { return Triangle(); }
    Segment segment(Face_handle f, int i) const { return Segment(); }
    Segment segment(const Edge& e) const { return Segment(); }
    Segment segment(const Edge_circulator& ec) const { return Segment(); }
    Segment segment(const All_edges_iterator& ei) const { return Segment(); }
    Segment segment(const Finite_edges_iterator& ei) const { return Segment(); }
    Point circumcenter(Face_handle f) const { return Point(); }
    Point circumcenter(const Point& p0,
		       const Point& p1,
		       const Point& p2) const { return Point(); }

    void flip(Face_handle f, int i) {};
    Vertex_handle insert_first(const Point& p) { return Vertex_handle(); }
    Vertex_handle insert_second(const Point& p){ return Vertex_handle(); }
    Vertex_handle insert_in_edge(const Point& p, Face_handle f,int i){ return Vertex_handle(); }
    Vertex_handle insert_in_face(const Point& p, Face_handle f){ return Vertex_handle(); }
    Vertex_handle insert_outside_convex_hull(const Point& p, Face_handle f){ return Vertex_handle(); }
    Vertex_handle insert_outside_affine_hull(const Point& p){ return Vertex_handle(); }
    Vertex_handle insert(const Point &p, Face_handle start = Face_handle() ){ return Vertex_handle(); }
    Vertex_handle insert(const Point& p,
			 Locate_type lt,
			 Face_handle loc, int li ){ return Vertex_handle(); }
    Vertex_handle push_back(const Point& a) { return Vertex_handle(); }
  
    void remove_degree_3(Vertex_handle  v, Face_handle f = Face_handle()) {};
    void remove_first(Vertex_handle  v) {};
    void remove_second(Vertex_handle v) {};
    void remove(Vertex_handle  v) {};

    Vertex_handle move_if_no_collision(Vertex_handle v, const Point &p) { return Vertex_handle(); }
    Vertex_handle move(Vertex_handle v, const Point &p) { return Vertex_handle(); }

    Face_handle locate(const Point& p,
		       Locate_type& lt,
		       int& li,
		       Face_handle start = Face_handle()) const { return Face_handle(); }
    Face_handle locate(const Point &p,
		       Face_handle start = Face_handle()) const { return Face_handle(); }

    Finite_faces_iterator finite_faces_begin() const { return _tds.faces_begin(); }
    Finite_faces_iterator finite_faces_end() const { return _tds.faces_end(); }
    Finite_vertices_iterator finite_vertices_begin() const { return _tds.vertices_begin(); }
    Finite_vertices_iterator finite_vertices_end() const { return _tds.vertices_end(); }
    Finite_edges_iterator finite_edges_begin() const { return _tds.edges_begin(); }
    Finite_edges_iterator finite_edges_end() const { return _tds.edges_end(); }
  
    All_faces_iterator all_faces_begin() const { return _tds.faces_begin(); }
    All_faces_iterator all_faces_end() const { return _tds.faces_end(); }
    All_vertices_iterator all_vertices_begin() const { return _tds.vertices_begin(); }
    All_vertices_iterator all_vertices_end() const { return _tds.vertices_end(); }
    All_edges_iterator all_edges_begin() const { return _tds.edges_begin(); }
    All_edges_iterator all_edges_end() const { return _tds.edges_end(); }
  
    Face_iterator faces_begin() const {return finite_faces_begin();}
    Face_iterator faces_end() const {return finite_faces_end();}
    Edge_iterator edges_begin() const {return finite_edges_begin();}
    Edge_iterator edges_end() const {return finite_edges_end();}
    Vertex_iterator vertices_begin() const {return finite_vertices_begin();}
    Vertex_iterator vertices_end() const {return finite_vertices_end();}
  
    Face_circulator incident_faces(Vertex_handle v,
				   Face_handle f =  Face_handle()) const { return Face_circulator(); }
    Vertex_circulator incident_vertices(Vertex_handle v,
					Face_handle f = Face_handle()) const { return Vertex_circulator(); }
    Edge_circulator incident_edges(Vertex_handle v,
				   Face_handle f = Face_handle()) const { return Edge_circulator(); }

    Vertex_handle mirror_vertex(Face_handle f, int i) const { return Vertex_handle(); }
    int mirror_index(Face_handle f, int i) const { return 0; }
    Edge mirror_edge(const Edge e) const { return Edge(); }

    Oriented_side
    oriented_side(const Point &p0, const Point &p1,
		  const Point &p2, const Point &p) const { return ON_ORIENTED_BOUNDARY; }
    Bounded_side
    bounded_side(const Point &p0, const Point &p1,
		 const Point &p2, const Point &p) const { return ON_BOUNDARY; }
    Oriented_side
    oriented_side(Face_handle f, const Point &p) const { return ON_ORIENTED_BOUNDARY; }
    Oriented_side
    side_of_oriented_circle(const Point &p0, const Point &p1, const Point &p2,
			    const Point &p, bool perturb) const { return ON_ORIENTED_BOUNDARY; }
    Oriented_side
    side_of_oriented_circle(Face_handle f, const Point & p, bool perturb = false) const { return ON_ORIENTED_BOUNDARY; }

    void make_hole(Vertex_handle v, std::list<Edge> & hole) {};
    Face_handle create_face(const Face& f = Face()) { return Face_handle(); }
    Face_handle create_face(Face_handle f) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2,
			    Face_handle f3, int i3) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1,
			    Face_handle f2, int i2) { return Face_handle(); }
    Face_handle create_face(Face_handle f1, int i1, Vertex_handle v) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3) { return Face_handle(); }
    Face_handle create_face(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3,
			    Face_handle f1,
			    Face_handle f2,
			    Face_handle f3) { return Face_handle(); }
    Face_handle create_face() { return Face_handle(); }
    void delete_face(Face_handle f) {};
    void delete_vertex(Vertex_handle v) {};

    template < class InputIterator >
    std::ptrdiff_t insert(InputIterator first, InputIterator last) { return 0; }
    // typedef typename std::vector< std::pair<Point,Info> >::iterator pinsert;
    // void insert(pinsert p1, pinsert p2) {};

    template<class EdgeIt>
    Vertex_handle star_hole( const Point& p,
			     EdgeIt edge_begin,
			     EdgeIt edge_end) { return Vertex_handle(); }
    template<class EdgeIt, class FaceIt>
    Vertex_handle star_hole( const Point& p,
			     EdgeIt edge_begin,
			     EdgeIt edge_end,
			     FaceIt face_begin,
			     FaceIt face_end) { return Vertex_handle(); }


    // Delaunay stuff
    int ccw(int i) const { return i; }
    int cw(int i) const { return i; }
    Point dual (Face_handle f) const { return Point(); }
    // Point circumcenter (Face_handle f) const { return Point(); }
    Vertex_handle nearest_vertex(Point p, Face_handle f = Face_handle()) const { return Vertex_handle(); }
    void flip_flippable(Face_handle x, int i) {}

    template <class OutputItFaces, class OutputItBoundaryEdges>
    std::pair<OutputItFaces,OutputItBoundaryEdges>
    get_conflicts_and_boundary(const Point  &p,
			       OutputItFaces fit,
			       OutputItBoundaryEdges eit,
			       Face_handle start = Face_handle(),
			       bool strict = true) const { return std::make_pair(fit, eit); }
    template <class OutputItFaces>
    OutputItFaces get_conflicts (const Point  &p,
				 OutputItFaces fit,
				 Face_handle start= Face_handle(),
				 bool strict = true) const { return fit; }
    template <class OutputItBoundaryEdges>
    OutputItBoundaryEdges
    get_boundary_of_conflicts(const Point  &p,
			      OutputItBoundaryEdges eit,
			      Face_handle start= Face_handle(),
			      bool strict = true) const { return eit; }
  };

  // Things specific to 3D case
  template <class Tds>
  class Triangulation_ds_edge_iterator_3
  {
  public:
    typedef typename Tds::Edge           Edge;
    typedef typename Tds::Cell_iterator  Cell_iterator;
    typedef typename Tds::Cell_handle    Cell_handle;
    typedef Triangulation_ds_edge_iterator_3<Tds> Edge_iterator;
  private:
    const Tds* _tds;
    Cell_iterator pos;
    mutable Edge edge;
  public:
    typedef Edge            value_type;
    typedef Edge*           pointer;
    typedef Edge&           reference;
    Triangulation_ds_edge_iterator_3() {};
    bool operator==(const Edge_iterator& fi) const { return false; }
    bool operator!=(const Edge_iterator& fi) const { return false; }
    Edge_iterator& operator++() { return *this; }
    Edge_iterator& operator--() { return *this; }
    Edge_iterator operator++(int) {  
      Edge_iterator tmp(*this);
      ++(*this);
      return tmp;
    }
    Edge_iterator operator--(int) {
      Edge_iterator tmp(*this);
      --(*this);
      return tmp;
    }
    Edge* operator->() const {
      edge.first = pos;
      return &edge;
    }
    Edge& operator*() const {
      edge.first = pos;
      return edge;
    }
  };

  template <class Tds>
  class Triangulation_ds_facet_iterator_3
  {
  public:
    typedef typename Tds::Facet          Facet;
    typedef typename Tds::Cell_iterator  Cell_iterator;
    typedef Triangulation_ds_facet_iterator_3<Tds> Facet_iterator;
  private:
    const Tds* _tds;
    Cell_iterator pos;
    mutable Facet facet;
  public:
    typedef Facet            value_type;
    typedef Facet*           pointer;
    typedef Facet&           reference;
    Triangulation_ds_facet_iterator_3() {};
    bool operator==(const Facet_iterator& fi) const { return false; }
    bool operator!=(const Facet_iterator& fi) const { return false; }
    Facet_iterator& operator++() { return *this; }
    Facet_iterator& operator--() { return *this; }
    Facet_iterator operator++(int) {  
      Facet_iterator tmp(*this);
      ++(*this);
      return tmp;
    }
    Facet_iterator operator--(int) {
      Facet_iterator tmp(*this);
      --(*this);
      return tmp;
    }
    Facet* operator->() const {
      facet.first = pos;
      return &facet;
    }
    Facet& operator*() const {
      facet.first = pos;
      return facet;
    }
  };

  template < class Tds>
  class Triangulation_ds_cell_circulator_3
  {
  public:
    typedef Triangulation_ds_cell_circulator_3<Tds> Cell_circulator;
    typedef typename Tds::Cell                      Cell;
    typedef typename Tds::Vertex                    Vertex;
    typedef typename Tds::Cell_handle               Cell_handle;
    typedef typename Tds::Vertex_handle             Vertex_handle;
  private:
    Vertex_handle _v;
    Cell_handle    pos;
  public:
    Triangulation_ds_cell_circulator_3() 
      : _v(), pos() 
    {}
    Triangulation_ds_cell_circulator_3(Vertex_handle v,
				       Cell_handle f = Cell_handle()) {
      _v = v;
      pos = f;
    }
    Cell_circulator& operator=(const Cell_circulator& other) { return *this; }
    Cell_circulator& operator++() { return *this; }
    Cell_circulator operator++(int) {
      Cell_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    Cell_circulator& operator--() { return *this; }
    Cell_circulator operator--(int) {
      Cell_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    operator Cell_handle() const { return pos; }
    bool operator==(const int i) const { return false; }
    bool operator!=(const int i) const { return false; }
    bool operator==(const Cell_circulator &fc) const { return false; }
    bool operator!=(const Cell_circulator &fc) const { return false; }
    bool operator==(const Cell_handle &fh) const { return pos == fh; }
    bool operator!=(const Cell_handle &fh) const { return pos != fh; }
    Cell& operator*() const { return *pos; }
    Cell* operator->() const { return &*pos; }
  };
  template < class Tds_ >
  bool operator==(typename Tds_::Cell_handle fh,
		  Triangulation_ds_cell_circulator_3<Tds_> fc) { return (fc==fh); }
  template < class Tds_ >
  bool operator!=(typename Tds_::Cell_handle fh,
		  Triangulation_ds_cell_circulator_3<Tds_> fc) { return (fc!=fh); }

  template < class Tds >
  class Triangulation_ds_facet_circulator_3 {
  public:
    typedef Triangulation_ds_facet_circulator_3<Tds> Facet_circulator;
    typedef typename Tds::Cell                       Cell;
    typedef typename Tds::Vertex                     Vertex;
    typedef typename Tds::Edge                       Edge;
    typedef typename Tds::Facet                      Facet;
    typedef typename Tds::Cell_handle                Cell_handle;
    typedef typename Tds::Vertex_handle              Vertex_handle;

  private:
    Vertex_handle _s;
    Vertex_handle _t;
    Cell_handle  pos;

  public:
    Triangulation_ds_facet_circulator_3()
      : _s(), _t(), pos()
    {}
    Triangulation_ds_facet_circulator_3(Cell_handle c, int s, int t)
      : _s(c->vertex(s)), _t(c->vertex(t)), pos(c)
    {}
    Triangulation_ds_facet_circulator_3(const Edge & e)
      : _s(e.first->vertex(e.second)), _t(e.first->vertex(e.third)), pos(e.first)
    {}
    Triangulation_ds_facet_circulator_3(Cell_handle c, int s, int t,
					Cell_handle start, int f)
      : _s(c->vertex(s)), _t(c->vertex(t))
    {}
    Triangulation_ds_facet_circulator_3(Cell_handle c, int s, int t,
					const Facet & start)
      : _s(c->vertex(s)), _t(c->vertex(t))
    {}
    Triangulation_ds_facet_circulator_3(const Edge & e, Cell_handle start, int f)
      : _s(e.first->vertex(e.second)), _t(e.first->vertex(e.third))
    {}
    Triangulation_ds_facet_circulator_3(const Edge & e, const Facet & start)
      : _s(e.first->vertex(e.second)), _t(e.first->vertex(e.third))
    {}

    Facet_circulator& operator++() { return *this; }
    Facet_circulator operator++(int) {
      Facet_circulator tmp(*this);
      ++(*this);
      return tmp;
    }
    Facet_circulator& operator--() { return *this; }
    Facet_circulator operator--(int) {
      Facet_circulator tmp(*this);
      --(*this);
      return tmp;
    }
    bool operator==(const int i) const { return false; }
    bool operator!=(const int i) const { return false; }
    bool operator==(const Facet_circulator &vc) const { return false; }
    bool operator!=(const Facet_circulator &vc) const { return false; }
    bool is_empty() const { return false; }
    Facet operator*() const { return Facet(); }
    struct Proxy_Facet {
      Proxy_Facet(const Facet & ff) : f(ff) {}
      Facet f;
      const Facet* operator->() const { return &f; }
    };
    Proxy_Facet operator->() const { return Proxy_Facet(* *this); }
  };

  template < typename Info_, typename GT>
  class Triangulation_vertex_base_with_info_3 {
    Info_ _info;
  public:
    typedef Info_ Info;
  };

  template < typename GT>
  class Triangulation_cell_base_with_circumcenter_3 {};

  template < class Vb, class Cb >
  class Triangulation_data_structure_3 {
  public:
    typedef Triangulation_data_structure_3<Vb,Cb>      Tds;
    typedef typename Vb::Info                          Info;
    typedef Vertex_base_3<Tds,Info>                     Vertex;
    typedef Cell_base<Tds>                             Cell;
    
  private:
    friend class Triangulation_ds_facet_iterator_3<Tds>;
    friend class Triangulation_ds_edge_iterator_3<Tds>;
    friend class Triangulation_ds_cell_circulator_3<Tds>;
    friend class Triangulation_ds_facet_circulator_3<Tds>;

  public:
    typedef Triangulation_ds_facet_iterator_3<Tds>     Facet_iterator;
    typedef Triangulation_ds_edge_iterator_3<Tds>      Edge_iterator;
    typedef Triangulation_ds_cell_circulator_3<Tds>    Cell_circulator;
    typedef Triangulation_ds_facet_circulator_3<Tds>   Facet_circulator;
    
    typedef Container<Vertex>                          Vertex_range;
    typedef Container<Cell>                            Cell_range;
    typedef typename Vertex_range::iterator            Vertex_iterator;
    typedef typename Cell_range::iterator              Cell_iterator;
    typedef Cell_iterator                              Cell_handle;
    typedef Vertex_iterator                            Vertex_handle;
    typedef std::pair<Cell_handle, int>                Facet;
    typedef Triple<Cell_handle, int, int>              Edge;

  protected:
    Cell_range   _cells;
    Vertex_range _vertices;

  public:
    Triangulation_data_structure_3() {};
    Cell_range& cells() { return _cells; }
    Cell_range& cells() const { return const_cast<Tds*>(this)->_cells; }
    Vertex_range& vertices() { return _vertices; }
    Vertex_range& vertices() const { return const_cast<Tds*>(this)->_vertices; }

    int dimension() const { return 0; }
    int number_of_vertices() const { return 0; }
    int number_of_cells() const { return 0; }
    int number_of_edges() const { return 0; }
    int number_of_facets() const { return 0; }
    int number_of_full_dim_faces() const { return 0; }
    
    bool is_valid(bool verbose = false, int level = 0) const { return false; }
    bool is_valid(Vertex_handle v, bool verbose = false, int level = 0) const { return false; }
    bool is_valid(Cell_handle c, bool verbose = false, int level = 0) const { return false; }

    bool is_vertex(Vertex_handle v) const { return false; }
    bool is_edge(Cell_handle c, int i, int j) const { return false; }
    bool is_edge(Vertex_handle u, Vertex_handle v, Cell_handle & c,
		 int & i, int & j) const { return false; }
    bool is_edge(Vertex_handle u, Vertex_handle v) const { return false; }
    bool is_facet(Cell_handle c, int i) const { return false; }
    bool is_facet(Vertex_handle u, Vertex_handle v,
		  Vertex_handle w,
		  Cell_handle & c, int & i, int & j, int & k) const { return false; }
    bool is_cell(Cell_handle c) const { return false; }
    bool is_cell(Vertex_handle u, Vertex_handle v,
		 Vertex_handle w, Vertex_handle t,
		 Cell_handle & c, int & i, int & j, int & k, int & l) const { return false; }
    bool is_cell(Vertex_handle u, Vertex_handle v,
		 Vertex_handle w, Vertex_handle t) const { return false; }
    bool has_vertex(const Facet & f, Vertex_handle v, int & j) const { return false; }
    bool has_vertex(Cell_handle c, int i,
		    Vertex_handle v, int & j) const { return false; }
    bool has_vertex(const Facet & f, Vertex_handle v) const { return false; }
    bool has_vertex(Cell_handle c, int i, Vertex_handle v) const { return false; }
  
    bool are_equal(Cell_handle c, int i,
		   Cell_handle n, int j) const { return false; }
    bool are_equal(const Facet & f, const Facet & g) const { return false; }
    bool are_equal(const Facet & f, Cell_handle n, int j) const { return false; }

    Cell_iterator cells_begin() const { return cells_end(); }
    Cell_iterator cells_end() const { return cells().end(); }
    Facet_iterator facets_begin() const { return Facet_iterator(); }
    Facet_iterator facets_end() const { return Facet_iterator(); }
    Edge_iterator edges_begin() const { return Edge_iterator(); }
    Edge_iterator edges_end() const { return Edge_iterator(); }
    Vertex_iterator vertices_begin() const { return vertices_end(); }
    Vertex_iterator vertices_end() const { return vertices().end(); }

    Cell_circulator incident_cells(const Edge & e) const { return Cell_circulator(); }
    Cell_circulator incident_cells(Cell_handle ce, int i, int j) const { return Cell_circulator(); }
    Cell_circulator incident_cells(const Edge &e, Cell_handle start) const { return Cell_circulator(); }
    Cell_circulator incident_cells(Cell_handle ce, int i, int j,
				   Cell_handle start) const { return Cell_circulator(); }
    Facet_circulator incident_facets(const Edge & e) const { return Facet_circulator(); }
    Facet_circulator incident_facets(Cell_handle ce, int i, int j) const { return Facet_circulator(); }
    Facet_circulator incident_facets(const Edge & e, const Facet & start) const { return Facet_circulator(); }
    Facet_circulator incident_facets(Cell_handle ce, int i, int j,
				     const Facet & start) const { return Facet_circulator(); }
    Facet_circulator incident_facets(const Edge & e,
				     Cell_handle start, int f) const { return Facet_circulator(); }
    Facet_circulator incident_facets(Cell_handle ce, int i, int j,
				     Cell_handle start, int f) const { return Facet_circulator(); }
    template <class OutputIterator>
    OutputIterator incident_cells(Vertex_handle v, OutputIterator cells) const { return cells; }
    template <class OutputIterator>
    OutputIterator incident_facets(Vertex_handle v, OutputIterator facets) const { return facets; }
    template <class Filter, class OutputIterator>
    OutputIterator incident_edges(Vertex_handle v, OutputIterator edges, Filter f = Filter()) const { return edges; }
    template <class OutputIterator>
    OutputIterator incident_edges(Vertex_handle v, OutputIterator edges) const { return edges; }
    template <class OutputIterator>
    OutputIterator incident_vertices(Vertex_handle v, OutputIterator vertices) const { return vertices; }
    template <class OutputIterator>
    OutputIterator adjacent_vertices(Vertex_handle v, OutputIterator vertices) const { return vertices; }
    
    bool flip(Cell_handle c, int i) { return false; }
    bool flip(const Facet &f) { return false; }
    void flip_flippable(Cell_handle c, int i) {};
    void flip_flippable(const Facet &f) {};
    bool flip(Cell_handle c, int i, int j) { return false; }
    bool flip(const Edge &e) { return false; }
    void flip_flippable(Cell_handle c, int i, int j) {}
    void flip_flippable(const Edge &e) {}

    Vertex_handle insert_first() { return Vertex_handle(); }
    Vertex_handle insert_second() { return Vertex_handle(); }
    Vertex_handle insert_in_cell(Cell_handle f) { return Vertex_handle(); }
    Vertex_handle insert_in_facet(const Facet & f) { return Vertex_handle(); }
    Vertex_handle insert_in_facet(Cell_handle f, int i) { return Vertex_handle(); }
    Vertex_handle insert_in_edge(const Edge & e) { return Vertex_handle(); }
    Vertex_handle insert_in_edge(Cell_handle f, int i, int j) { return Vertex_handle(); }

    Vertex_handle create_vertex(const Vertex &v = Vertex()) { return Vertex_handle(); }
    Vertex_handle create_vertex(Vertex_handle v) { return Vertex_handle(); }
    Cell_handle create_cell(const Cell& f = Cell()) { return Cell_handle(); }
    Cell_handle create_cell(Cell_handle f) { return Cell_handle(); }
    Cell_handle create_cell(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3,
			    Vertex_handle v4) { return Cell_handle(); }
    Cell_handle create_cell(Vertex_handle v1,
			    Vertex_handle v2,
			    Vertex_handle v3,
			    Vertex_handle v4,
			    Cell_handle f1,
			    Cell_handle f2,
			    Cell_handle f3,
			    Cell_handle f4) { return Cell_handle(); }
    Cell_handle create_face() { return Cell_handle(); }
    Cell_handle create_face(Vertex_handle v1,
                            Vertex_handle v2,
                            Vertex_handle v3) { return Cell_handle(); }
    Cell_handle create_face(Cell_handle f1, int i1,
			    Cell_handle f2, int i2,
			    Cell_handle f3, int i3) { return Cell_handle(); }
    Cell_handle create_face(Cell_handle f1, int i1,
			    Cell_handle f2, int i2) { return Cell_handle(); }
    Cell_handle create_face(Cell_handle f1, int i1, Vertex_handle v) { return Cell_handle(); }
    void set_adjacency(Cell_handle f0, int i0, Cell_handle f1, int i1) const {};
    void delete_cell(Cell_handle) {};
    void delete_vertex(Vertex_handle) {};

    void clear() {};
    void set_dimension (int n) {};

  };

  template < class Gt, class Tds >
  class Delaunay_triangulation_3 {
  public:

    typedef typename Tds::Vertex                 Vertex;
    typedef typename Tds::Cell                   Cell;
    typedef typename Tds::Facet                  Facet;
    typedef typename Tds::Edge                   Edge;

    typedef typename Tds::Vertex_handle          Vertex_handle;
    typedef typename Tds::Cell_handle            Cell_handle;

    typedef typename Tds::Cell_circulator        Cell_circulator;
    typedef typename Tds::Facet_circulator       Facet_circulator;

    typedef typename Tds::Cell_iterator          Cell_iterator;
    typedef typename Tds::Facet_iterator         Facet_iterator;
    typedef typename Tds::Edge_iterator          Edge_iterator;
    typedef typename Tds::Vertex_iterator        Vertex_iterator;
  
    typedef Cell_iterator                        All_cells_iterator;
    typedef Facet_iterator                       All_facets_iterator;
    typedef Edge_iterator                        All_edges_iterator;
    typedef Vertex_iterator                      All_vertices_iterator;

    typedef Filter_iterator<All_cells_iterator>  Finite_cells_iterator;
    typedef Filter_iterator<All_facets_iterator> Finite_facets_iterator;
    typedef Filter_iterator<All_edges_iterator>  Finite_edges_iterator;
    typedef Filter_iterator<All_vertices_iterator>  Finite_vertices_iterator;

    enum Locate_type {VERTEX=0,
		      EDGE,
		      FACE,
		      OUTSIDE_CONVEX_HULL,
		      OUTSIDE_AFFINE_HULL};

    typedef typename Tds::Info Info;
    typedef typename Gt::Point Point;
    typedef typename Gt::Triangle Triangle;
    typedef typename Gt::Tetrahedron Tetrahedron;
    typedef typename Gt::Segment Segment;

  protected:
    Gt  _gt;
    Tds _tds;
    Vertex_handle _infinite_vertex;

  public:
    Delaunay_triangulation_3() {};
    void clear() {};
    const Tds & tds() const { return _tds;}
    Tds & tds() { return _tds;}
    bool is_valid() const { return false; }
    int dimension() const { return 0; }
    int number_of_vertices() const { return 0; }
    int number_of_edges() const { return 0; }
    int number_of_facets() const { return 0; }
    int number_of_cells() const { return 0; }
    int number_of_finite_edges() const { return 0; }
    int number_of_finite_facets() const { return 0; }
    int number_of_finite_cells() const { return 0; }
    Vertex_handle infinite_vertex() const { return _infinite_vertex; }
    Cell_handle infinite_cell() const { return Cell_handle(); }
    void set_infinite_vertex(const Vertex_handle& v) {};

    bool is_infinite(Vertex_handle v) const { return false; }
    bool is_infinite(Cell_handle f) const { return false; }
    bool is_infinite(Cell_handle f, int i) const { return false; }
    bool is_infinite(const Facet & f) const { return false; }
    bool is_infinite(const Cell_handle c, int i, int j) const { return false; }
    bool is_infinite(const Edge& e) const { return false; }

    bool is_vertex(Vertex_handle v) const { return false; }
    bool is_edge(Vertex_handle u, Vertex_handle v, Cell_handle & c,
		 int & i, int & j) const { return false; }
    bool is_facet(Vertex_handle u, Vertex_handle v,
		  Vertex_handle w,
		  Cell_handle & c, int & i, int & j, int & k) const { return false; }
    bool is_cell(Cell_handle c) const { return false; }
    bool is_cell(Vertex_handle u, Vertex_handle v,
		 Vertex_handle w, Vertex_handle t,
		 Cell_handle & c, int & i, int & j, int & k, int & l) const { return false; }
    bool is_cell(Vertex_handle u, Vertex_handle v,
		 Vertex_handle w, Vertex_handle t) const { return false; }
    bool has_vertex(const Facet & f, Vertex_handle v, int & j) const { return false; }
    bool has_vertex(Cell_handle c, int i,
		    Vertex_handle v, int & j) const { return false; }
    bool has_vertex(const Facet & f, Vertex_handle v) const { return false; }
    bool has_vertex(Cell_handle c, int i, Vertex_handle v) const { return false; }
  
    bool are_equal(Cell_handle c, int i,
		   Cell_handle n, int j) const { return false; }
    bool are_equal(const Facet & f, const Facet & g) const { return false; }
    bool are_equal(const Facet & f, Cell_handle n, int j) const { return false; }

    Finite_cells_iterator finite_cells_begin() const { return _tds.cells_begin(); }
    Finite_cells_iterator finite_cells_end() const { return _tds.cells_end(); }
    Finite_vertices_iterator finite_vertices_begin() const { return _tds.vertices_begin(); }
    Finite_vertices_iterator finite_vertices_end() const { return _tds.vertices_end(); }
    Finite_edges_iterator finite_edges_begin() const { return _tds.edges_begin(); }
    Finite_edges_iterator finite_edges_end() const { return _tds.edges_end(); }
    Finite_facets_iterator finite_facets_begin() const { return _tds.facets_begin(); }
    Finite_facets_iterator finite_facets_end() const { return _tds.facets_end(); }
  
    All_cells_iterator all_cells_begin() const { return _tds.cells_begin(); }
    All_cells_iterator all_cells_end() const { return _tds.cells_end(); }
    All_vertices_iterator all_vertices_begin() const { return _tds.vertices_begin(); }
    All_vertices_iterator all_vertices_end() const { return _tds.vertices_end(); }
    All_edges_iterator all_edges_begin() const { return _tds.edges_begin(); }
    All_edges_iterator all_edges_end() const { return _tds.edges_end(); }
    All_facets_iterator all_facets_begin() const { return _tds.facets_begin(); }
    All_facets_iterator all_facets_end() const { return _tds.facets_end(); }
  
    Cell_iterator cells_begin() const {return finite_cells_begin();}
    Cell_iterator cells_end() const {return finite_cells_end();}
    Edge_iterator edges_begin() const {return finite_edges_begin();}
    Edge_iterator edges_end() const {return finite_edges_end();}
    Vertex_iterator vertices_begin() const {return finite_vertices_begin();}
    Vertex_iterator vertices_end() const {return finite_vertices_end();}
    Facet_iterator facets_begin() const {return finite_facets_begin();}
    Facet_iterator facets_end() const {return finite_facets_end();}
  
    Cell_circulator incident_cells(const Edge & e) const { return Cell_circulator(); }
    Cell_circulator incident_cells(Cell_handle ce, int i, int j) const { return Cell_circulator(); }
    Cell_circulator incident_cells(const Edge &e, Cell_handle start) const { return Cell_circulator(); }
    Cell_circulator incident_cells(Cell_handle ce, int i, int j,
				   Cell_handle start) const { return Cell_circulator(); }
    Facet_circulator incident_facets(const Edge & e) const { return Facet_circulator(); }
    Facet_circulator incident_facets(Cell_handle ce, int i, int j) const { return Facet_circulator(); }
    Facet_circulator incident_facets(const Edge & e, const Facet & start) const { return Facet_circulator(); }
    Facet_circulator incident_facets(Cell_handle ce, int i, int j,
				     const Facet & start) const { return Facet_circulator(); }
    Facet_circulator incident_facets(const Edge & e,
				     Cell_handle start, int f) const { return Facet_circulator(); }
    Facet_circulator incident_facets(Cell_handle ce, int i, int j,
				     Cell_handle start, int f) const { return Facet_circulator(); }
    template <class OutputIterator>
    OutputIterator incident_cells(Vertex_handle v, OutputIterator cells) const { return cells; }
    template <class OutputIterator>
    OutputIterator incident_facets(Vertex_handle v, OutputIterator facets) const { return facets; }
    template <class Filter, class OutputIterator>
    OutputIterator incident_edges(Vertex_handle v, OutputIterator edges, Filter f = Filter()) const { return edges; }
    template <class OutputIterator>
    OutputIterator incident_edges(Vertex_handle v, OutputIterator edges) const { return edges; }
    template <class OutputIterator>
    OutputIterator incident_vertices(Vertex_handle v, OutputIterator vertices) const { return vertices; }
    template <class OutputIterator>
    OutputIterator adjacent_vertices(Vertex_handle v, OutputIterator vertices) const { return vertices; }

    void remove(Vertex_handle v) {};
    Vertex_handle move_if_no_collision(Vertex_handle v, const Point &p) { return Vertex_handle(); }
    Vertex_handle move(Vertex_handle v, const Point &p) { return Vertex_handle(); }

    Bounded_side side_of_sphere(Cell_handle c, const Point & p,
				bool perturb = false) const { return ON_BOUNDARY; }
    Bounded_side side_of_circle( const Facet & f, const Point & p, bool perturb = false) const { return ON_BOUNDARY; }
    Bounded_side side_of_circle( Cell_handle c, int i, const Point & p,
				 bool perturb = false) const { return ON_BOUNDARY; }

    Tetrahedron tetrahedron(const Cell_handle c) const { return Tetrahedron(); }
    Triangle triangle(const Cell_handle c, int i) const { return Triangle(); }
    Triangle triangle(const Facet & f) const { return Triangle(); }
    Segment segment(Cell_handle f, int i1, int i2) const { return Segment(); }
    Segment segment(const Edge& e) const { return Segment(); }

    Cell_handle locate(const Point & p,
		       Locate_type & lt, int & li, int & lj,
		       Cell_handle start = Cell_handle()) const { return start; }
    Cell_handle locate(const Point & p, Cell_handle start = Cell_handle()) const { return start; }
    Cell_handle locate(const Point & p,
		       Locate_type & lt, int & li, int & lj, Vertex_handle hint) const { return Cell_handle(); }
    Cell_handle locate(const Point & p, Vertex_handle hint) const { return Cell_handle(); }
    
    int mirror_index(Cell_handle c, int i) const { return 0; }
    Vertex_handle mirror_vertex(Cell_handle c, int i) const { return Vertex_handle(); }
    Facet mirror_facet(Facet f) const { return Facet(); }

    bool flip(const Facet &f) { return false; }
    bool flip(Cell_handle c, int i) { return false; }
    void flip_flippable(const Facet &f) {};
    void flip_flippable(Cell_handle c, int i) {};
    bool flip(const Edge &e) { return false; }
    bool flip(Cell_handle c, int i, int j) { return false; }
    void flip_flippable(const Edge &e) {};
    void flip_flippable(Cell_handle c, int i, int j) {};

    Vertex_handle insert(const Point & p, Vertex_handle hint) { return hint; }
    Vertex_handle insert(const Point & p, Cell_handle start = Cell_handle()) { return Vertex_handle(); }
    Vertex_handle insert(const Point & p, Locate_type lt, Cell_handle c,
			 int li, int lj) { return Vertex_handle(); }
    template < class InputIterator >
    std::ptrdiff_t insert(InputIterator first, InputIterator last) { return 0; }
    Vertex_handle insert_in_cell(const Point & p, Cell_handle c) { return Vertex_handle(); }
    Vertex_handle insert_in_facet(const Point & p, Cell_handle c, int i) { return Vertex_handle(); }
    Vertex_handle insert_in_facet(const Point & p, const Facet & f) { return Vertex_handle(); }
    Vertex_handle insert_in_edge(const Point & p, Cell_handle c, int i, int j) { return Vertex_handle(); }
    Vertex_handle insert_in_edge(const Point & p, const Edge & e) { return Vertex_handle(); }
    Vertex_handle insert_outside_convex_hull(const Point & p, Cell_handle c) { return Vertex_handle(); }
    Vertex_handle insert_outside_affine_hull(const Point & p) { return Vertex_handle(); }

    // Delaunay stuff
    int ccw(int i) const { return i; }
    int cw(int i) const { return i; }
    Vertex_handle nearest_vertex_in_cell(const Point& p, Cell_handle c) const { return Vertex_handle(); }
    Vertex_handle nearest_vertex(const Point& p, Cell_handle c = Cell_handle()) const { return Vertex_handle(); }
    bool is_Gabriel(Cell_handle c, int i) const { return false; }
    bool is_Gabriel(Cell_handle c, int i, int j) const { return false; }
    bool is_Gabriel(const Facet& f) const { return false; }
    bool is_Gabriel(const Edge& e) const { return false; }
    Point dual(Cell_handle c) const { return Point(); }
    bool is_valid(Cell_handle c, bool verbose = false, int level = 0) const { return false; }

    template <class OutputIteratorBoundaryFacets,
	      class OutputIteratorCells,
	      class OutputIteratorInternalFacets>
    Triple<OutputIteratorBoundaryFacets,
	   OutputIteratorCells,
	   OutputIteratorInternalFacets>
    find_conflicts(const Point &p, Cell_handle c,
		   OutputIteratorBoundaryFacets bfit,
		   OutputIteratorCells cit,
		   OutputIteratorInternalFacets ifit) const { 
      return Triple<OutputIteratorBoundaryFacets,
		    OutputIteratorCells,
		    OutputIteratorInternalFacets>(bfit, cit, ifit); 
    }
    template <class OutputIteratorBoundaryFacets, class OutputIteratorCells>
    std::pair<OutputIteratorBoundaryFacets, OutputIteratorCells>
    find_conflicts(const Point &p, Cell_handle c,
		   OutputIteratorBoundaryFacets bfit,
		   OutputIteratorCells cit) const { return std::make_pair(bfit, cit); }
    template <class OutputIterator>
    OutputIterator vertices_in_conflict(const Point&p, Cell_handle c, OutputIterator res) const { return res; }
    template <class OutputIterator>
    OutputIterator vertices_on_conflict_zone_boundary(const Point&p, Cell_handle c,
						      OutputIterator res) const { return res; }
    
  };

};
