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
void swap_cells(I *verts, I *neigh, uint32_t ndim, uint64_t i1, uint64_t i2) {
  I t;
  uint32_t d;
  for (d = 0; d < ndim; d++) {
    t = verts[i1*ndim + d];
    verts[i1*ndim + d] = verts[i2*ndim + d];
    verts[i2*ndim + d] = t;
    t = neigh[i1*ndim + d];
    neigh[i1*ndim + d] = neigh[i2*ndim + d];
    neigh[i2*ndim + d] = t;
  }
}

template <typename I>
class CellMap
{
public:
  std::map<std::vector<I>, uint64_t> _m;
  uint32_t ndim;
  typename std::map<std::vector<I>, uint64_t> map_type;
  CellMap() {}
  CellMap(uint32_t _ndim) {
    ndim = _ndim;
  }
  CellMap(uint32_t _ndim, uint64_t ninit, I *v, uint64_t *idx) {
    ndim = _ndim;
    uint64_t i;
    uint32_t j;
    std::vector<I> key;
    for (j = 0; j < (ndim+1); j++)
      key.push_back(0);
    for (i = 0; i < ninit; i++) {
      for (j = 0; j < ndim+1; j++)
	key[j] = v[i*(ndim+1)+j];
      _m.insert(std::make_pair(key, idx[i]));
    }
  }
  ~CellMap() {}
  uint64_t size() {
    return (uint64_t)(_m.size());
  }
  template <typename leafI>
  uint64_t insert(leafI *v, uint32_t *sort_v, uint64_t new_idx) {
    std::vector<I> key;
    for (uint32_t i = 0; i < (ndim+1); i++)
      key.push_back((I)(v[sort_v[i]]));
    uint64_t idx = _m.insert(std::make_pair(key, new_idx)).first->second;
    return idx;
  }
  template <typename leafI>
  uint64_t insert_long(leafI *v, leafI *sort_v, uint64_t new_idx) {
    std::vector<I> key;
    for (uint32_t i = 0; i < (ndim+1); i++)
      key.push_back((I)(v[sort_v[i]]));
    uint64_t idx = _m.insert(std::make_pair(key, new_idx)).first->second;
    return idx;
  }
  template <typename leafI>
  void put_in_arrays(leafI *keys, uint64_t *vals) {
    typename std::map<std::vector<I>, uint64_t>::iterator it;
    std::vector<I> ikey;
    uint64_t i;
    uint32_t j;
    i = 0;
    for (it = _m.begin(); it != _m.end(); it++) {
      ikey = it->first;
      for (j = 0; j < (ndim+1); j++)
	keys[i*(ndim+1)+j] = (leafI)(ikey[j]);
      vals[i] = it->second;
      i++;
    }
  }
};

std::size_t findtype_SerializedLeaf(const char* filename) {
  std::ifstream os(filename, std::ios::binary);
  if (!os) std::cerr << "Error cannot open file: " << filename << std::endl;
  else {
    std::size_t size_I;
    os.read((char*)&size_I, sizeof(std::size_t));
    os.close();
    return size_I;
  }
  return 0;
}

template <typename I>
class SerializedLeaf
{
public:
  int id;
  uint32_t ndim;
  int64_t ncells;
  uint64_t idx_start;
  uint64_t idx_stop;
  I idx_inf;
  I *verts;
  I *neigh;
  uint32_t *sort_verts;
  uint64_t *sort_cells;
  int64_t *visited;
  bool init_from_file;
  SerializedLeaf() {
    visited = NULL;
    init_from_file = false;
  }
  SerializedLeaf(int _id, uint32_t _ndim, int64_t _ncells, I _idx_inf,
		 I *_verts, I *_neigh,
		 uint32_t *_sort_verts, uint64_t *_sort_cells,
		 uint64_t _idx_start, uint64_t _idx_stop) {
    id = _id;
    ndim = _ndim;
    ncells = _ncells;
    idx_inf = _idx_inf;
    verts = _verts;
    neigh = _neigh;
    sort_verts = _sort_verts;
    sort_cells = _sort_cells;
    idx_start = _idx_start;
    idx_stop = _idx_stop;
    visited = (int64_t*)malloc(ncells*sizeof(int64_t));
    for (int64_t i = 0; i < ncells; i++)
      visited[i] = -1;
    init_from_file = false;
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

  void write_to_file(const char* filename) const {
    std::ofstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot create file: " << filename << std::endl;
    else {
      int64_t i;
      uint32_t j;
      std::size_t size_I = sizeof(I);

      // Write header
      os.write((char*)&size_I, sizeof(std::size_t));
      os.write((char*)&id, sizeof(int));
      os.write((char*)&ndim, sizeof(uint32_t));
      os.write((char*)&ncells, sizeof(int64_t));
      os.write((char*)&idx_inf, size_I);
      os.write((char*)&idx_start, sizeof(uint64_t));
      os.write((char*)&idx_stop, sizeof(uint64_t));
      if (ncells==0) {
        os.close();
        return;
      }
      // Write cell vertices
      for (i = 0; i < ncells; i++) {
	for (j = 0; j < (ndim+1); j++) {
	  os.write((char*)(verts+i*(ndim+1)+j), size_I);
	}
      }
      // Write cell neighbors
      for (i = 0; i < ncells; i++) {
	for (j = 0; j < (ndim+1); j++) {
	  os.write((char*)(neigh+i*(ndim+1)+j), size_I);
	}
      }
      // Write sorted verts
      for (i = 0; i < ncells; i++) {
	for (j = 0; j < (ndim+1); j++) {
	  os.write((char*)(sort_verts+i*(ndim+1)+j), sizeof(uint32_t));
	}
      }
      // Write sorted cells
      for (i = 0; i < ncells; i++) {
	os.write((char*)(sort_cells+i), sizeof(uint64_t));
      }
      os.close();

    }
  }

  int64_t read_from_file(const char* filename) {
    std::ifstream os(filename, std::ios::binary);
    if (!os) std::cerr << "Error cannot open file: " << filename << std::endl;
    else {
      int64_t i;
      uint32_t j;
      std::size_t size_I;

      // Read header
      os.read((char*)&size_I, sizeof(std::size_t));
      if (size_I != sizeof(I)) {
	std::cerr << "Error this object uses the wrong template type." << std::endl;
	os.close();
	return 0;
      }
      os.read((char*)&id, sizeof(int));
      os.read((char*)&ndim, sizeof(uint32_t));
      os.read((char*)&ncells, sizeof(int64_t));
      os.read((char*)&idx_inf, size_I);
      os.read((char*)&idx_start, sizeof(uint64_t));
      os.read((char*)&idx_stop, sizeof(uint64_t));
      if (ncells==0) {
        os.close();
        return 0;
      }
      // Read cell vertices
      verts = (I*)malloc(ncells*(ndim+1)*size_I);
      for (i = 0; i < ncells; i++) {
	for (j = 0; j < (ndim+1); j++) {
	  os.read((char*)(verts+i*(ndim+1)+j), size_I);
	}
      }
      // Read cell neighbors
      neigh = (I*)malloc(ncells*(ndim+1)*size_I);
      for (i = 0; i < ncells; i++) {
	for (j = 0; j < (ndim+1); j++) {
	  os.read((char*)(neigh+i*(ndim+1)+j), size_I);
	}
      }
      // Read sorted verts
      sort_verts = (uint32_t*)malloc(ncells*(ndim+1)*sizeof(uint32_t));
      for (i = 0; i < ncells; i++) {
	for (j = 0; j < (ndim+1); j++) {
	  os.read((char*)(sort_verts+i*(ndim+1)+j), sizeof(uint32_t));
	}
      }
      // Read sorted cells
      sort_cells = (uint64_t*)malloc(ncells*sizeof(uint64_t));
      for (i = 0; i < ncells; i++) {
	os.read((char*)(sort_cells+i), sizeof(uint64_t));
      }
      os.close();

      // Create visited array
      init_from_file = true;
      visited = (int64_t*)malloc(ncells*sizeof(int64_t));
      for (int64_t i = 0; i < ncells; i++)
	visited[i] = -1;

      return ncells;
    }
    return 0;
  }

  void cleanup() {
    if (visited != NULL) 
      free(visited);
    if (init_from_file) {
      free(verts);
      free(neigh);
      free(sort_verts);
      free(sort_cells);
    }
  }

};

template <typename I>
class ConsolidatedLeaves
{
public:
  uint32_t ndim;
  int64_t ncells;
  int64_t max_ncells;
  I *allverts;
  I *allneigh;
  I idx_inf;
  CellMap<I> split_map;
  CellMap<I> inf_map;
  uint32_t *matches1;
  uint32_t *matches2;
  ConsolidatedLeaves() {}
  ConsolidatedLeaves(uint32_t _ndim, I _idx_inf, int64_t _max_ncells, 
		     I *_verts, I *_neigh) {
    ncells = 0;
    max_ncells = _max_ncells;
    ndim = _ndim;
    allverts = _verts;
    allneigh = _neigh;
    idx_inf = _idx_inf;
    split_map = CellMap<I>(ndim);
    inf_map = CellMap<I>(ndim);
    matches1 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
    matches2 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
  }
  ConsolidatedLeaves(uint32_t _ndim, int64_t _ncells, I _idx_inf,
		     int64_t _max_ncells, I *_verts, I *_neigh) {
    ncells = _ncells;
    max_ncells = _max_ncells;
    ndim = _ndim;
    allverts = _verts;
    allneigh = _neigh;
    idx_inf = _idx_inf;
    split_map = CellMap<I>(ndim);
    inf_map = CellMap<I>(ndim);
    matches1 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
    matches2 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
  }
  ConsolidatedLeaves(uint32_t _ndim, int64_t _ncells, I _idx_inf,
		     int64_t _max_ncells, I *_verts, I *_neigh,
		     uint64_t n_split_map, I *key_split_map, uint64_t *val_split_map,
		     uint64_t n_inf_map, I *key_inf_map, uint64_t *val_inf_map) {
    // This version allows use of pre-existing information
    ncells = _ncells;
    max_ncells = _max_ncells;
    ndim = _ndim;
    allverts = _verts;
    allneigh = _neigh;
    idx_inf = _idx_inf;
    split_map = CellMap<I>(ndim, n_split_map, key_split_map, val_split_map);
    inf_map = CellMap<I>(ndim, n_inf_map, key_inf_map, val_inf_map);
    matches1 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
    matches2 = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
  }
  uint64_t size_split_map() {
    return split_map.size();
  }
  uint64_t size_inf_map() {
    return inf_map.size(); 
  }
  void get_split_map(I *keys, uint64_t *vals) {
    split_map.put_in_arrays(keys, vals);
  }
  void get_inf_map(I *keys, uint64_t *vals) {
    inf_map.put_in_arrays(keys, vals);
  }

  void cleanup() {
    // printf("%lu cells in split_map, %lu cells in inf_map.\n",split_map.size(),inf_map.size());
    free(matches1);
    free(matches2);
  }

  void add_leaf_fromfile(const char* filename) {
    std::size_t size_I;
    size_I = findtype_SerializedLeaf(filename);
    if (size_I == 4) {
      SerializedLeaf<uint32_t> leaf = SerializedLeaf<uint32_t>();
      leaf.read_from_file(filename);
      add_leaf(leaf);
      leaf.cleanup();
    } else if (size_I == 8) {
      SerializedLeaf<uint64_t> leaf = SerializedLeaf<uint64_t>();
      leaf.read_from_file(filename);
      add_leaf(leaf);
      leaf.cleanup();
    } else {
      std::cerr << "Error this file does not follow a recognized template type. Size = " << size_I << std::endl;
      return;
    }
  }

  template <typename leafI>
  void add_leaf(SerializedLeaf<leafI> leaf) {
    leafI i;
    int64_t idx;
    for (i = 0; i < (leafI)leaf.ncells; i++) {
      idx = add_cell(leaf, i);
      if (idx >= 0) {
      	add_neigh(leaf, i, idx);
      }
    }
  }

  template <typename leafI>
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

  template <typename leafI>
  int64_t add_cell(SerializedLeaf<leafI> leaf, leafI icell) {
    int64_t idx = -1;
    leafI *verts = leaf.verts + icell*(ndim+1);
    uint32_t *sort_verts = leaf.sort_verts + icell*(ndim+1);
    if (leaf.visited[icell] != -1)
      return leaf.visited[icell];
    // Infinite cell, skip for now
    // if (verts[sort_verts[0]] == leaf.idx_inf) {
    //   leaf.visited[icell] = -777;
    //   return idx;
    // }
    // Finite
    leafI vmax = verts[sort_verts[0]];
    leafI vmin = verts[sort_verts[ndim]];
    if (vmax == leaf.idx_inf)
      vmax = verts[sort_verts[1]];
    if ((vmin >= leaf.idx_start) and (vmax < leaf.idx_stop)) {
      // All points on this leaf
      idx = leaf.visited[icell];
      if (idx < 0) {
	idx = new_cell(verts);
	leaf.visited[icell] = idx;
      }
    } else {
      // Points split between leaves
      idx = split_map.insert(verts, sort_verts, (uint64_t)(ncells));
      leaf.visited[icell] = idx;
      if (idx == ncells) 
	idx = new_cell(verts);
    }

    return idx;
  }

  template <typename leafI>
  void add_neigh(SerializedLeaf<leafI> leaf, leafI icell, int64_t c_total) {
    bool match;
    uint32_t n_local, n_total, n_other, n;
    int64_t c_other;
    I c_exist;
    leafI *verts = leaf.verts + icell*(ndim+1);
    leafI *neigh = leaf.neigh + icell*(ndim+1);
    for (n_local = 0; n_local < (ndim+1); n_local++) {
      if (neigh[n_local] == leaf.idx_inf)
	continue;
      c_other = leaf.visited[neigh[n_local]];
      if (c_other < 0)
	continue;
      if (verts[n_local] == leaf.idx_inf) {
	for (n_total = 0; n_total < (ndim+1); n_total++) {
	  if (allverts[c_total*(ndim+1)+n_total] == idx_inf)
	    break;
	}
      } else {
	for (n_total = 0; n_total < (ndim+1); n_total++) {
	  if (allverts[c_total*(ndim+1)+n_total] == verts[n_local])
	    break;
	}
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
      } else if (c_exist != (I)c_other) {
	// New neighbor does not match existing one
	printf("There are conflicting neighbors for cell %ld on leaf %ld.\n",
	       (int64_t)(icell), (int64_t)(leaf.id));
	int i;
	uint32_t *sort_verts = (uint32_t*)malloc((ndim+1)*sizeof(uint32_t));
	std::vector<int> src_leaves;
	// This leaf
	for (i = 0; i < (int)(ndim+1); i++) sort_verts[i] = i;
	arg_sortCellVerts(allverts+c_total*(ndim+1), sort_verts, 1, ndim+1);
	// src_leaves = find_leaves(allverts+c_total*(ndim+1), sort_verts);
	printf("    this (%ld): ", c_total);
	for (i = 0; i < (int)(ndim+1); i++) 
	  printf("%ld ", (int64_t)(allverts[c_total*(ndim+1)+i]));
	// printf(", leaves = ");
	// for (i = 0; i < (int)(src_leaves.size()); i++)
	//   printf("%d ", src_leaves[i]);
	printf("\n");
	// The existing neighbor
	for (i = 0; i < (int)(ndim+1); i++) sort_verts[i] = i;
	arg_sortCellVerts(allverts+c_exist*(ndim+1), sort_verts, 1, ndim+1);
	// src_leaves = find_leaves(allverts+c_exist*(ndim+1), sort_verts);
	printf("    old (%ld): ", (int64_t)(c_exist));
	for (i = 0; i < (int)(ndim+1); i++) 
	  printf("%ld ", (int64_t)(allverts[c_exist*(ndim+1)+i]));
	// printf(", leaves = ");
	// for (i = 0; i < (int)(src_leaves.size()); i++)
	//   printf("%d ", src_leaves[i]);
	printf("\n");
	// The new neighbor
	for (i = 0; i < (int)(ndim+1); i++) sort_verts[i] = i;
	arg_sortCellVerts(allverts+c_other*(ndim+1), sort_verts, 1, ndim+1);
	// src_leaves = find_leaves(allverts+c_other*(ndim+1), sort_verts);
	printf("    new (%ld): ", c_other);
	for (i = 0; i < (int)(ndim+1); i++) 
	  printf("%ld ", (int64_t)(allverts[c_other*(ndim+1)+i]));
	// printf(", leaves = ");
	// for (i = 0; i < (int)(src_leaves.size()); i++)
	//   printf("%d ", src_leaves[i]);
	printf("\n");
	// // throw std::logic_error;
      }
    }
  }

  template<typename I2>
  std::vector<int> find_leaves(uint64_t num_leaves, 
			       uint64_t *leaf_start_idx, uint64_t *leaf_stop_idx,
			       I2 *verts, uint32_t *sort_verts) {
    std::vector<int> out;
    int ic_final = 0, il_final = num_leaves;
    int ic = ndim, il = 0;
    I2 x, sidx;
    while ((ic >= ic_final) and (il < il_final)) {
      sidx = leaf_start_idx[il];
      while ((ic >= ic_final) and (verts[sort_verts[ic]] < sidx)) 
	ic--;
      if (ic < ic_final)
	break;
      x = verts[sort_verts[ic]];
      while ((il < il_final) and (x >= leaf_stop_idx[il]))
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

  int64_t count_inf() {
    int64_t out = 0;
    int64_t c;
    for (c = 0; c < (ncells*(ndim+1)); c++) {
      if (allneigh[c] == idx_inf)
	out++;
    }
    return out;
  }

  void add_inf() {
    if ((ncells + count_inf()) > max_ncells) {
      printf("Adding infinite cells (%ld) will exceed maximum (%ld).\n",
	     count_inf(), max_ncells);
      return;
    }
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
	  // Add to map
	  for (n = 0; n < (ndim+1); n++)
	    sort_verts[n] = (I)(n);
	  arg_quickSort(new_verts, sort_verts, 1, 0, ndim);
	  idx = inf_map.insert_long(new_verts, sort_verts, (uint64_t)(ncells));
	  neigh[idx_miss] = idx;
	  if (idx == ncells) {
	    append_cell(new_verts, new_neigh, norig, ncells);
	  }
	}
      }
    }
    free(new_verts);
    free(new_neigh);
    free(sort_verts);
    free(idx_fwd);
  }
  
};

