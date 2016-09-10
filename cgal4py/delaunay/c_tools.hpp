#include <vector>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>

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

