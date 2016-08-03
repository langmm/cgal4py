#include <vector>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include "../c_utils.hpp"

class Node
{
public:
  bool is_leaf;
  uint32_t ndim;
  std::vector<double> left_edge;
  std::vector<double> right_edge;
  uint64_t left_idx;
  uint64_t children;
  // innernode parameters
  uint32_t split_dim;
  double split;
  Node *less;
  Node *greater;
  // innernode constructor
  Node(uint32_t ndim0, std::vector<double> le, std::vector<double> re, uint64_t Lidx, 
       uint32_t sdim0, double split0, Node *lnode, Node *gnode)
  {
    is_leaf = false;
    ndim = ndim0;
    left_edge = le;
    right_edge = re;
    left_idx = Lidx;

    split_dim = sdim0;
    split = split0;
    less = lnode;
    greater = gnode;
    children = lnode->children + gnode->children;
  }
  // leafnode constructor
  Node(uint32_t ndim0, std::vector<double> le, std::vector<double> re, 
       uint64_t Lidx, uint64_t n)
  {
    is_leaf = true;
    ndim = ndim0;
    left_edge = le;
    right_edge = re;
    left_idx = Lidx;

    children = n;
  }
};

class KDTree
{
public:
  double* all_pts;
  uint64_t* all_idx;
  uint64_t npts;
  uint32_t ndim;
  uint32_t leafsize;
  double* domain_left_edge;
  double* domain_right_edge;
  double* domain_mins;
  double* domain_maxs;
  std::vector<Node*> leaves;
  Node* root;

  // KDTree() {}
  KDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m, uint32_t leafsize0, 
	 double *left_edge, double *right_edge)
  {
    all_pts = pts;
    all_idx = idx;
    npts = n;
    ndim = m;
    leafsize = leafsize0;
    domain_left_edge = left_edge;
    domain_right_edge = right_edge;

    domain_mins = min_pts(pts, n, m);
    domain_maxs = max_pts(pts, n, m);

    std::vector<double> LE;
    std::vector<double> RE;
    std::vector<double> mins;
    std::vector<double> maxs;
    for (uint32_t d = 0; d < m; d++) {
      LE.push_back(left_edge[d]);
      RE.push_back(right_edge[d]);
      mins.push_back(domain_mins[d]);
      maxs.push_back(domain_maxs[d]);
    }

    root = build(0, n, LE, RE, mins, maxs);
  }
  ~KDTree()
  {
    // free(idx);
    free(domain_mins);
    free(domain_maxs);
    free(root);
  }

  Node* build(uint64_t Lidx, uint64_t n, 
	      std::vector<double> LE, std::vector<double> RE, 
	      std::vector<double> mins, std::vector<double> maxes)
  {
    if (n < leafsize) {
      Node* out = new Node(ndim, LE, RE, Lidx, n);
      leaves.push_back(out);
      return out;
    } else {
      // Find dimension to split along
      uint32_t dmax, d;
      dmax = 0;
      for (d = 1; d < ndim; d++) 
	if ((maxes[d]-mins[d]) > (maxes[dmax]-mins[dmax]))
	  dmax = d;
      if (maxes[dmax] == mins[dmax]) {
	// all points singular
	Node* out = new Node(ndim, LE, RE, Lidx, n);
	leaves.push_back(out);
	return out;
      }
      
      // Find median along dimension
      int64_t stop = n-1;
      int64_t med = (n/2) + (n%2);

      // Version using pointer to all points and index
      med = select(all_pts, all_idx, ndim, dmax, Lidx, stop+Lidx, (stop/2)+Lidx);
      med = (stop/2)+Lidx;
      uint64_t Nless = med-Lidx+1;
      uint64_t Ngreater = n - Nless;
      double split;
      if ((n%2) == 0) {
	split = all_pts[ndim*all_idx[med] + dmax];
	// split = (all_pts[ndim*all_idx[med] + dmax] + 
	// 	 all_pts[ndim*all_idx[med+1] + dmax])/2.0;
      } else {
	split = all_pts[ndim*all_idx[med] + dmax];
      }

      // Get new boundaries
      std::vector<double> lessmaxes;
      std::vector<double> lessright;
      std::vector<double> greatermins;
      std::vector<double> greaterleft;
      for (d = 0; d < ndim; d++) {
	lessmaxes.push_back(maxes[d]);
	lessright.push_back(RE[d]);
	greatermins.push_back(mins[d]);
	greaterleft.push_back(LE[d]);
      }
      lessmaxes[dmax] = split;
      lessright[dmax] = split;
      greatermins[dmax] = split;
      greaterleft[dmax] = split;

      // Build less and greater nodes
      Node* less = build(Lidx, Nless, LE, lessright, mins, lessmaxes);
      Node* greater = build(Lidx+Nless, Ngreater, greaterleft, RE, greatermins, maxes);

      // Create innernode referencing child nodes
      Node* out = new Node(ndim, LE, RE, Lidx, dmax, split, less, greater);
      return out;
    } 
  }	 
};


