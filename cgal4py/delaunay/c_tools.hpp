#include <vector>
#include <map>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <exception>

bool intersect_sph_box(uint32_t ndim, double *c, double r, double *le, double *re) {
  uint32_t i;
  for (i = 0; i < ndim; i++) {
    if (c[i] < le[i]) {
      if ((c[i] + r) < le[i])
	return false;
    } else if (c[i] > re[i]) {
      if ((c[i] - r) > re[i]) 
	return false;
    } 
  }
  return true;
}

template<typename I>
bool arg_tLT(I *cells, uint32_t *idx_verts, uint32_t ndim, uint64_t i1, uint64_t i2)
{
  uint32_t d;
  for (d = 0; d < ndim; d++) {
    if (cells[i1*ndim+idx_verts[i1*ndim+d]] < cells[i2*ndim+idx_verts[i2*ndim+d]])
      return true;
    else if (cells[i1*ndim+idx_verts[i1*ndim+d]] > cells[i2*ndim+idx_verts[i2*ndim+d]])
      return false;
  }
  // Equal
  return false;
}

template<typename I>
bool tEQ(I *cells, uint32_t ndim, int64_t i1, int64_t i2)
{
  uint32_t d;
  for (d = 0; d < ndim; d++) {
    if (cells[i1*ndim+d] != cells[i2*ndim+d])
      return false;
  }
  // Equal
  return true;

}

template<typename I>
bool tGT(I *cells, uint32_t ndim, int64_t i1, int64_t i2)
{
  uint32_t d;
  for (d = 0; d < ndim; d++) {
    if (cells[i1*ndim+d] > cells[i2*ndim+d])
      return true;
    else if (cells[i1*ndim+d] < cells[i2*ndim+d])
      return false;
  }
  // Equal
  return false;
}

template<typename I>
bool tLT(I *cells, uint32_t ndim, int64_t i1, int64_t i2)
{
  uint32_t d;
  for (d = 0; d < ndim; d++) {
    if (cells[i1*ndim+d] < cells[i2*ndim+d])
      return true;
    else if (cells[i1*ndim+d] > cells[i2*ndim+d])
      return false;
  }
  // Equal
  return false;
}

template<typename I>
void swapCells(I *cells, I *neigh, I *idx, uint32_t ndim, 
	       int64_t i1, int64_t i2)
{
  int64_t k;
  I tc, tn;
  for (k = 0; k < ndim; k++) {
    tc = cells[i1*ndim+k];
    tn = neigh[i1*ndim+k];
    cells[i1*ndim+k] = cells[i2*ndim+k];
    neigh[i1*ndim+k] = neigh[i2*ndim+k];
    cells[i2*ndim+k] = tc;
    neigh[i2*ndim+k] = tn;
  }
  tc = idx[i1]; idx[i1] = idx[i2]; idx[i2] = tc;
}

template<typename I>
int64_t arg_partition(I *arr, I *idx, uint32_t ndim,
		      int64_t l, int64_t r, int64_t p)
{ 
  int64_t i, j;
  I t;

  // Put pivot element in lowest element
  t = idx[p]; idx[p] = idx[l]; idx[l] = t;
  p = l;

  for (i = l+1, j = r; i <= j; ) {
    if ((tLT(arr, ndim, idx[p], idx[i])) && (not tLT(arr, ndim, idx[p], idx[j]))) {
      t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    if (not tLT(arr, ndim, idx[p], idx[i])) i++;
    if (tLT(arr, ndim, idx[p], idx[j])) j--;
  }

  // Put pivot element at j
  t = idx[l]; idx[l] = idx[j]; idx[j] = t;

  return j;
}

template<typename I>
void arg_quickSort(I *arr, I *idx, uint32_t ndim,
		   int64_t l, int64_t r)
{
  int64_t j;
  if ( l < r )
    {
      j = arg_partition(arr, idx, ndim, l, r, (l+r)/2);
      arg_quickSort(arr, idx, ndim, l, j-1);
      arg_quickSort(arr, idx, ndim, j+1, r);
    }
}

template<typename I>
int64_t partition_tess(I *cells, I *neigh, I *idx, uint32_t ndim,
		       int64_t l, int64_t r, int64_t p)
{ 
  int64_t i, j;

  // Put pivot element in lowest element
  swapCells(cells, neigh, idx, ndim, p, l);
  p = l;

  for (i = l+1, j = r; i <= j; ) {
    if ((tLT(cells, ndim, p, i)) && (not tLT(cells, ndim, p, j))) {
      swapCells(cells, neigh, idx, ndim, i, j);
    }
    if (not tLT(cells, ndim, p, i)) i++;
    if (tLT(cells, ndim, p, j)) j--;
  }

  // Put pivot element at j
  swapCells(cells, neigh, idx, ndim, l, j);

  return j;
}

template<typename I>
void quickSort_tess(I *cells, I *neigh, I *idx, uint32_t ndim,
		    int64_t l, int64_t r)
{
  int64_t j;
  if ( l < r )
    {
      j = partition_tess(cells, neigh, idx, ndim, l, r, (l+r)/2);
      quickSort_tess(cells, neigh, idx, ndim, l, j-1);
      quickSort_tess(cells, neigh, idx, ndim, j+1, r);
    }
}

template<typename I>
void sortCellVerts(I *cells, I *neigh, uint64_t ncells, uint32_t ndim)
{
  int64_t i, j, c;
  I tc, tn;
  int64_t l;
  int64_t r;

  for (c = 0; c < (int64_t)ncells; c++) {
    l = ndim*c;
    r = l + ndim - 1;

    if (l == r) continue;
    for (i = l+1; i <= r; i++) {
      tc = cells[i];
      tn = neigh[i];
      j = i - 1;
      while ((j >= l) && (cells[j] < tc)) {
	cells[j+1] = cells[j];
	neigh[j+1] = neigh[j];
	j--;
      }
      cells[j+1] = tc;
      neigh[j+1] = tn;
    }
  }
}

template<typename I>
void sortSerializedTess(I *cells, I *neigh, 
			uint64_t ncells, uint32_t ndim)
{
  // Sort vertices in each cell w/ neighbors
  sortCellVerts(cells, neigh, ncells, ndim);
  // Get indices to sort cells
  I *idx_fwd = (I*)malloc(ncells*sizeof(I));
  I *idx_rev = (I*)malloc(ncells*sizeof(I));
  for (I i = 0; i < (I)ncells; i++) {
    idx_fwd[i] = i;
    idx_rev[i] = i;
  }
  quickSort_tess(cells, neigh, idx_fwd, ndim, 0, ncells-1);
  arg_quickSort(idx_fwd, idx_rev, 1, 0, ncells-1);
  // Sort cells w/ neighbors
  for (I i = 0; i < (I)(ncells*ndim); i++) {
    if (neigh[i] < (I)ncells)
      neigh[i] = idx_rev[neigh[i]];
  }
  // quickSort_tess(cells, neigh, ndim, 0, ncells-1);

  free(idx_fwd);
  free(idx_rev);
}

// Version only getting indexes of sort
template<typename I>
int64_t arg_partition_tess(I *cells, uint32_t *idx_verts, uint64_t *idx_cells,
			   uint32_t ndim, int64_t l, int64_t r, int64_t p)
{ 
  uint64_t t;
  int64_t i, j;

  // Put pivot element in lowest element
  t = idx_cells[l]; idx_cells[l] = idx_cells[p]; idx_cells[p] = t;
  p = l;

  for (i = l+1, j = r; i <= j; ) {
    if ((arg_tLT(cells, idx_verts, ndim, idx_cells[p], idx_cells[i])) && 
	(not arg_tLT(cells, idx_verts, ndim, idx_cells[p], idx_cells[j]))) {
      t = idx_cells[i]; idx_cells[i] = idx_cells[j]; idx_cells[j] = t;
    }
    if (not arg_tLT(cells, idx_verts, ndim, idx_cells[p], idx_cells[i])) i++;
    if (arg_tLT(cells, idx_verts, ndim, idx_cells[p], idx_cells[j])) j--;
  }

  // Put pivot element at j
  t = idx_cells[l]; idx_cells[l] = idx_cells[j]; idx_cells[j] = t;

  return j;
}

template<typename I>
void arg_quickSort_tess(I *cells, uint32_t *idx_verts, uint64_t *idx_cells, 
			uint32_t ndim, int64_t l, int64_t r)
{
  int64_t j;
  if ( l < r )
    {
      j = arg_partition_tess(cells, idx_verts, idx_cells, ndim, l, r, (l+r)/2);
      arg_quickSort_tess(cells, idx_verts, idx_cells, ndim, l, j-1);
      arg_quickSort_tess(cells, idx_verts, idx_cells, ndim, j+1, r);
    }
}

template<typename I>
void arg_sortCellVerts(I *cells, uint32_t *idx, uint64_t ncells, uint32_t ndim)
{
  int64_t i, j, c;
  I t;
  int64_t l;
  int64_t r;

  for (c = 0; c < (int64_t)ncells; c++) {
    l = ndim*c;
    r = l + ndim - 1;

    if (l == r) continue;
    for (i = l+1; i <= r; i++) {
      t = idx[i];
      j = i - 1;
      while ((j >= l) && (cells[l+idx[j]] < cells[l+t])) {
	idx[j+1] = idx[j];
	j--;
      }
      idx[j+1] = t;
    }
  }
}

template<typename I>
void arg_sortSerializedTess(I *cells, uint64_t ncells, uint32_t ndim, 
			    uint32_t *idx_verts, uint64_t *idx_cells)
{
  // Sort vertices in each cell w/ neighbors
  arg_sortCellVerts(cells, idx_verts, ncells, ndim);
  // Get indices to sort cells
  arg_quickSort_tess(cells, idx_verts, idx_cells, ndim, 0, ncells-1);
}

template <typename I>
class CellMap
{
public:
  std::map<std::vector<I>, int64_t> _m;
  uint32_t ndim;
  CellMap() {}
  CellMap(uint32_t _ndim) {
    ndim = _ndim;
  }
  ~CellMap() {}
  int64_t insert(I *v, uint32_t *sort_v, int64_t new_idx) {
    std::vector<I> key;
    for (uint32_t i = 0; i < (ndim+1); i++)
      key.push_back(v[sort_v[i]]);
    int64_t idx = _m.insert(std::make_pair(key, new_idx)).first->second;
    return idx;
  }
  int64_t insert_long(I *v, I *sort_v, int64_t new_idx) {
    std::vector<I> key;
    for (uint32_t i = 0; i < (ndim+1); i++)
      key.push_back(v[sort_v[i]]);
    int64_t idx = _m.insert(std::make_pair(key, new_idx)).first->second;
    return idx;
  }
};

template <typename I>
class SerializedLeaf
{
public:
  int id;
  uint32_t ndim;
  int64_t ncells;
  I start_idx;
  I stop_idx;
  I idx_inf;
  I *verts;
  I *neigh;
  uint32_t *sort_verts;
  uint64_t *sort_cells;
  int64_t *visited;
  SerializedLeaf() {}
  SerializedLeaf(int _id, uint32_t _ndim, int64_t _ncells, 
		 I _start_idx, I _stop_idx, I _idx_inf,
		 I *_verts, I *_neigh,
		 uint32_t *_sort_verts, uint64_t *_sort_cells) {
    id = _id;
    ndim = _ndim;
    ncells = _ncells;
    start_idx = _start_idx;
    stop_idx = _stop_idx;
    idx_inf = _idx_inf;
    verts = _verts;
    neigh = _neigh;
    sort_verts = _sort_verts;
    sort_cells = _sort_cells;
    visited = (int64_t*)malloc(ncells*sizeof(int64_t));
    for (int64_t i = 0; i < ncells; i++)
      visited[i] = -1;
  }
  // ~SerializedLeaf() {
  //   free(visited);
  // }

  int64_t find_cell(I *v, uint32_t *sort_v) {
    int64_t i = 0;
    uint64_t isort;
    uint32_t d = 0;
    uint32_t dsort;
    bool equal, quit = false;
    while (i < ncells) {
      isort = sort_cells[i];
      equal = true;
      for (d = 0; d < (ndim+1); d++) {
	dsort = sort_verts[isort*(ndim+1)+d];
	if (v[sort_v[d]] < verts[isort*(ndim+1)+dsort]) {
	  equal = false;
	  break;
	} else if (v[sort_v[d]] > verts[isort*(ndim+1)+dsort]) {
	  equal = false;
	  quit = true;
	  break;
	}
      }
      if (equal) 
	return (int64_t)isort;
      if (quit)
	break;
      else
	i++;
    }
    return -1;
  }
};

template <typename I, typename leafI>
class ConsolidatedLeaves
{
public:
  uint32_t ndim;
  int64_t ncells;
  int64_t max_ncells;
  uint64_t num_leaves;
  I *allverts;
  I *allneigh;
  I idx_inf;
  std::vector<SerializedLeaf<leafI>> leaves;
  CellMap<leafI> split_map;
  CellMap<I> inf_map;
  uint32_t *matches1;
  uint32_t *matches2;
  ConsolidatedLeaves() {}
  ConsolidatedLeaves(uint32_t _ndim, uint64_t _num_leaves, I _idx_inf,
		     int64_t _max_ncells, I *_verts, I *_neigh,
		     std::vector<SerializedLeaf<leafI>> _leaves) {
    ncells = 0;
    max_ncells = _max_ncells;
    ndim = _ndim;
    num_leaves = _num_leaves;
    allverts = _verts;
    allneigh = _neigh;
    idx_inf = _idx_inf;
    leaves = _leaves;
    split_map = CellMap<leafI>(ndim);
    inf_map = CellMap<I>(ndim);
    matches1 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
    matches2 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
    for (uint64_t i = 0; i < num_leaves; i++)
      add_leaf(leaves[i]);
    // for (uint64_t i = 0; i < num_leaves; i++)
    //   add_leaf_neigh(leaves[i]);
    add_inf();
    free(matches1);
    free(matches2);
  }

  void add_leaf(SerializedLeaf<leafI> leaf) {
    leafI i;
    int64_t idx;
    for (i = 0; i < (leafI)leaf.ncells; i++) {
      idx = add_cell(leaf, i);
      if (idx >= 0)
      	add_neigh(leaf, i, idx);
    }
  }

  // void add_leaf_neigh(SerializedLeaf<leafI> leaf) {
  //   leafI i;
  //   int64_t idx;
  //   for (i = 0; i < (leafI)leaf.ncells; i++) {
  //     idx = leaf.visited[i];
  //     if (idx >= 0)
  // 	add_neigh(leaf, i, idx);
  //     else if (idx == -777)
  // 	return;

  //   }
  // }

  int64_t new_cell(leafI *verts) {
    uint32_t j;
    int64_t idx = ncells;
    for (j = 0; j < (ndim+1); j++) 
      allverts[idx*(ndim+1)+j] = (I)(verts[j]);
    ncells++;
    return idx;
  }

  int64_t append_cell(I *verts, I *neigh, int64_t start, int64_t stop) {
    uint32_t j;
    int64_t idx = ncells;
    find_neigh(verts, neigh, start, stop);
    for (j = 0; j < (ndim+1); j++) {
      allverts[idx*(ndim+1)+j] = (I)(verts[j]);
      allneigh[idx*(ndim+1)+j] = (I)(neigh[j]);
    }
    ncells++;
    return idx;
  }

  int64_t add_cell(SerializedLeaf<leafI> leaf, leafI icell) {
    int64_t idx = -1;
    leafI *verts = leaf.verts + icell*(ndim+1);
    uint32_t *sort_verts = leaf.sort_verts + icell*(ndim+1);
    if (leaf.visited[icell] != -1)
      return leaf.visited[icell];
    // Infinite cell, skip for now
    if (verts[sort_verts[0]] == leaf.idx_inf) {
      leaf.visited[icell] = -777;
      return idx;
    }
    // Finite
    std::vector<int> src_leaves = find_leaves(verts, sort_verts);
    if (src_leaves.size() == 1) {
      // All points on a single leaf
      if (src_leaves[0] == leaf.id) {
	// This leaf
	idx = leaf.visited[icell];
	if (idx < 0) {
	  idx = new_cell(verts);
	  leaf.visited[icell] = idx;
	}
      } else {
	// Another leaf
	SerializedLeaf<leafI> oth_leaf = leaves[src_leaves[0]];
	int64_t oth_cell = oth_leaf.find_cell(verts, sort_verts);
	if (oth_cell >= 0) {
	  idx = oth_leaf.visited[oth_cell];
	  if (idx < 0) {
	    idx = new_cell(verts);
	    leaf.visited[icell] = idx;
	    oth_leaf.visited[oth_cell] = idx;
	  } else {
	    leaf.visited[icell] = idx;
	  }
	} else
	  idx = -1;
      }
    } else {
      // idx = split_map.insert(verts, sort_verts, ncells);
      // leaf.visited[icell] = idx;
      // if (idx == ncells) 
      // 	idx = new_cell(verts);
      bool leaf_contributes = false;
      int i;
      for (i = 0; i < (int)(src_leaves.size()); i++) {
      	if (src_leaves[i] == leaf.id) {
      	  leaf_contributes = true;
      	  break;
      	}
      }
      if (leaf_contributes) {
	// this leaf contributes to this cell, do split map
      	idx = split_map.insert(verts, sort_verts, ncells);
      	leaf.visited[icell] = idx;
      	if (idx == ncells) 
      	  idx = new_cell(verts);
      } else {
	// this leaf dosn't contribute to this cell, but may provide neighbors.
	// check to see if any of the other contributing leaves also have it.
	int64_t oth_cell;
	SerializedLeaf<leafI> oth_leaf;
	bool cell_found = false;
	for (i = 0; i < (int)(src_leaves.size()); i++) {
	  oth_leaf = leaves[src_leaves[i]];
	  oth_cell = oth_leaf.find_cell(verts, sort_verts);
	  if (oth_cell >= 0) {
	    if (cell_found) {
	      idx = leaf.visited[icell];
	      oth_leaf.visited[oth_cell] = idx;
	    } else {
	      idx = oth_leaf.visited[oth_cell];
	      if (idx < 0) {
		idx = split_map.insert(verts, sort_verts, ncells);
		if (idx == ncells) 
		  idx = new_cell(verts);
		oth_leaf.visited[oth_cell] = idx;
	      }
	      leaf.visited[icell] = idx;
	      cell_found = true;
	    }
	  }
	}
      }
    }

    return idx;
  }

  void add_neigh(SerializedLeaf<leafI> leaf, leafI icell, int64_t c_total) {
    bool match;
    uint32_t n_local, n_total, n_other, n;
    int64_t c_other;
    I c_exist;
    leafI *verts = leaf.verts + icell*(ndim+1);
    leafI *neigh = leaf.neigh + icell*(ndim+1);
    for (n_local = 0; n_local < (ndim+1); n_local++) {
      c_other = leaf.visited[neigh[n_local]];
      if (c_other < 0)
	continue;
      for (n_total = 0; n_total < (ndim+1); n_total++) {
	if (allverts[c_total*(ndim+1)+n_total] == verts[n_local])
	  break;
      }
      // if (allverts[c_total*(ndim+1)+n_total] != verts[n_local])
      // 	printf("Error!!!\n");
      c_exist = allneigh[c_total*(ndim+1)+n_total];
      if (c_exist == idx_inf) {
	// This neighbor does not yet exist
	for (n_other = 0; n_other < (ndim+1); n_other++) {
	  match = false;
	  for (n = 0; n < (ndim+1); n++) {
	    if (allverts[c_other*(ndim+1)+n_other] == allverts[c_total*(ndim+1)+n]) {
	      match = true;
	      break;
	    }
	  }
	  if (not match) 
	    break;
	}
	allneigh[c_total*(ndim+1)+n_total] = (I)c_other;
	allneigh[c_other*(ndim+1)+n_other] = (I)c_total;
      } else if (c_exist == (I)c_other) {
	// New neighbor matches existing one
      } else {
	// New neighbor does not match existing one
	printf("There are conflicting neighbors for cell %ld on leaf %ld.\n",
	       (int64_t)(icell), (int64_t)(leaf.id));
	int i;
	uint32_t *sort_verts = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
	std::vector<int> src_leaves;
	// This leaf
	for (i = 0; i < (int)(ndim+1); i++) sort_verts[i] = i;
	arg_sortCellVerts(allverts+c_total*(ndim+1), sort_verts, 1, ndim+1);
	src_leaves = find_leaves(allverts+c_total*(ndim+1), sort_verts);
	printf("    this (%ld): ", c_total);
	for (i = 0; i < (int)(ndim+1); i++) 
	  printf("%ld ", (int64_t)(allverts[c_total*(ndim+1)+i]));
	printf(", leaves = ");
	for (i = 0; i < (int)(src_leaves.size()); i++)
	  printf("%d ", src_leaves[i]);
	printf("\n");
	// The existing neighbor
	for (i = 0; i < (int)(ndim+1); i++) sort_verts[i] = i;
	arg_sortCellVerts(allverts+c_exist*(ndim+1), sort_verts, 1, ndim+1);
	src_leaves = find_leaves(allverts+c_exist*(ndim+1), sort_verts);
	printf("    old (%ld): ", (int64_t)(c_exist));
	for (i = 0; i < (int)(ndim+1); i++) 
	  printf("%ld ", (int64_t)(allverts[c_exist*(ndim+1)+i]));
	printf(", leaves = ");
	for (i = 0; i < (int)(src_leaves.size()); i++)
	  printf("%d ", src_leaves[i]);
	printf("\n");
	// The new neighbor
	for (i = 0; i < (int)(ndim+1); i++) sort_verts[i] = i;
	arg_sortCellVerts(allverts+c_other*(ndim+1), sort_verts, 1, ndim+1);
	src_leaves = find_leaves(allverts+c_other*(ndim+1), sort_verts);
	printf("    new (%ld): ", c_other);
	for (i = 0; i < (int)(ndim+1); i++) 
	  printf("%ld ", (int64_t)(allverts[c_other*(ndim+1)+i]));
	printf(", leaves = ");
	for (i = 0; i < (int)(src_leaves.size()); i++)
	  printf("%d ", src_leaves[i]);
	printf("\n");
	// // throw std::logic_error;
      }
    }
  }

  template<typename I2>
  std::vector<int> find_leaves(I2 *verts, uint32_t *sort_verts) {
    std::vector<int> out;
    int ic_final = 0, il_final = num_leaves;
    int ic = ndim, il = 0;
    I2 x, sidx;
    while ((ic >= ic_final) and (il < il_final)) {
      sidx = leaves[il].start_idx;
      while ((ic >= ic_final) and (verts[sort_verts[ic]] < sidx)) 
	ic--;
      if (ic < ic_final)
	break;
      x = verts[sort_verts[ic]];
      while ((il < il_final) and (x >= leaves[il].stop_idx))
	il++;
      if (il >= il_final)
	break;
      out.push_back(il);
      ic--;
      il++;
    }
    return out;
  }

  void find_neigh(I *c1, I *n1, int64_t start, int64_t stop) {
    int64_t i2;
    I *c2;
    I *n2;
    uint32_t N_matches;
    uint32_t v1, v2, v;
    for (i2 = start; i2 < stop; i2++) {
      c2 = allverts + i2*(ndim+1);
      n2 = allneigh + i2*(ndim+1);
      N_matches = 0;
      for (v = 0; v < (ndim+1); v++) {
	matches1[v] = ndim+1;
	matches2[v] = ndim+1;
      }
      for (v2 = 0; v2 < (ndim+1); v2++) {
	for (v1 = 0; v1 < (ndim+1); v1++) {
	  if (c1[v1] == c2[v2]) {
	    matches1[v1] = v2;
	    matches2[v2] = v1;
	    N_matches++;
	    break;
	  }
	}
      }
      if (N_matches == ndim) {
	for (v1 = 0; v1 < (ndim+1); v1++) {
	  if (matches1[v1] > ndim) 
	    break;
	}
	for (v2 = 0; v2 < (ndim+1); v2++) {
	  if (matches2[v2] > ndim)
	    break;
	}
	n1[v1] = (I)i2;
	n2[v2] = (I)ncells;
      }
    }
  }

  void add_inf() {
    int i;
    uint32_t n, Nneigh, idx_miss;
    int64_t c, norig = ncells, idx;
    I *verts;
    I *neigh;
    I *new_verts = (I*)malloc((ndim+1)*sizeof(I));
    I *new_neigh = (I*)malloc((ndim+1)*sizeof(I));
    I *sort_verts = (I*)malloc((ndim+1)*sizeof(I));
    uint32_t *idx_fwd = (uint32_t*)malloc(ndim*sizeof(uint32_t));
    for (c = 0; c < norig; c++) {
      Nneigh = 0;
      verts = allverts + c*(ndim+1);
      neigh = allneigh + c*(ndim+1);
      for (n = 0; n < (ndim+1); n++) {
	if (neigh[n] != idx_inf)
	  Nneigh++;
      }
      if (Nneigh <= ndim) {
	for (idx_miss = 0; idx_miss < (ndim+1); idx_miss++) {
	  if (neigh[idx_miss] != idx_inf)
	    continue;
	  // Vertices & neighbors of new cell
	  i = 0;
	  for (n = 0; n < idx_miss; n++, i++) {
	    idx_fwd[i] = n;
	    new_neigh[n] = idx_inf;
	  }
	  for (n = (idx_miss+1); n < (ndim+1); n++, i++) {
	    idx_fwd[i] = n;
	    new_neigh[n] = idx_inf;
	  }
	  for (n = 0; n < ndim; n++) 
	    new_verts[idx_fwd[n]] = verts[idx_fwd[ndim-(n+1)]];
	  new_verts[idx_miss] = idx_inf;
	  new_neigh[idx_miss] = c;
	  // 
	  for (n = 0; n < (ndim+1); n++)
	    sort_verts[n] = (I)(n);
	  arg_quickSort(new_verts, sort_verts, 1, 0, ndim);
	  idx = inf_map.insert_long(new_verts, sort_verts, ncells);
	  neigh[idx_miss] = idx;
	  if (idx == ncells)
	    append_cell(new_verts, new_neigh, norig, ncells);
	}
      }
    }
    free(new_verts);
    free(new_neigh);
    free(sort_verts);
    free(idx_fwd);
  }
  
};

