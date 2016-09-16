import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
import copy

@cython.boundscheck(False)
@cython.wraparound(False)
def py_arg_tLT(np.ndarray[np.int64_t, ndim=2] cells, 
               np.ndarray[np.uint32_t, ndim=2] idx_verts, int i1, int i2):
    r"""Determine if one cell is less than the other by comparing 
    the (sorted) vertices in each cell.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices 
            for the n cells in a m-dimensional triangulation.
        idx_verts (np.ndarray of uint32): (n, m+1) array of indices for 
            sorting the vertices in each cell.
        i1 (int): Index of the 1st cell in the comparison.
        i2 (int): Index of the 2nd cell in the comparison.

    Returns:
        bool: Truth of `cells[i1,idx_verts[i1,:]] < cells[i2,idx_verts[i2,:]]`.

    """
    assert(cells.shape[0] != 0)
    assert(i1 < cells.shape[0])
    assert(i2 < cells.shape[0])
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    cdef cbool out
    with nogil:
        out = arg_tLT[int64_t](&cells[0,0], &idx_verts[0,0], ndim, i1, i2)
    return <pybool>out

@cython.boundscheck(False)
@cython.wraparound(False)
def py_tEQ(np.ndarray[np.int64_t, ndim=2] cells, int i1, int i2):
    r"""Determine if one cell is equivalent to the other by comparing 
    the (sorted) vertices in each cell.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices 
            for the n cells in a m-dimensional triangulation.
        i1 (int): Index of the 1st cell in the comparison.
        i2 (int): Index of the 2nd cell in the comparison.

    Returns:
        bool: Truth of `cells[i1,:] == cells[i2,:]`.

    """
    assert(cells.shape[0] != 0)
    assert(i1 < cells.shape[0])
    assert(i2 < cells.shape[0])
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    cdef cbool out
    with nogil:
        out = tEQ[int64_t](&cells[0,0], ndim, i1, i2)
    return <pybool>out

@cython.boundscheck(False)
@cython.wraparound(False)
def py_tGT(np.ndarray[np.int64_t, ndim=2] cells, int i1, int i2):
    r"""Determine if one cell is greater than the other by comparing 
    the (sorted) vertices in each cell.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices 
            for the n cells in a m-dimensional triangulation.
        i1 (int): Index of the 1st cell in the comparison.
        i2 (int): Index of the 2nd cell in the comparison.

    Returns:
        bool: Truth of `cells[i1,:] > cells[i2,:]`.

    """
    assert(cells.shape[0] != 0)
    assert(i1 < cells.shape[0])
    assert(i2 < cells.shape[0])
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    cdef cbool out
    with nogil:
        out = tGT[int64_t](&cells[0,0], ndim, i1, i2)
    return <pybool>out

@cython.boundscheck(False)
@cython.wraparound(False)
def py_tLT(np.ndarray[np.int64_t, ndim=2] cells, int i1, int i2):
    r"""Determine if one cell is less than the other by comparing 
    the (sorted) vertices in each cell.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices 
            for the n cells in a m-dimensional triangulation.
        i1 (int): Index of the 1st cell in the comparison.
        i2 (int): Index of the 2nd cell in the comparison.

    Returns:
        bool: Truth of `cells[i1,:] < cells[i2,:]`.

    """
    assert(cells.shape[0] != 0)
    assert(i1 < cells.shape[0])
    assert(i2 < cells.shape[0])
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    cdef cbool out
    with nogil:
        out = tLT[int64_t](&cells[0,0], ndim, i1, i2)
    return <pybool>out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _sortCellVerts_int64(np.ndarray[np.int64_t, ndim=2] cells,
                               np.ndarray[np.int64_t, ndim=2] neigh):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        sortCellVerts[int64_t](&cells[0,0], &neigh[0,0], ncells, ndim)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _sortCellVerts_uint64(np.ndarray[np.uint64_t, ndim=2] cells,
                                np.ndarray[np.uint64_t, ndim=2] neigh):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        sortCellVerts[uint64_t](&cells[0,0], &neigh[0,0], ncells, ndim)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _sortCellVerts_int32(np.ndarray[np.int32_t, ndim=2] cells,
                               np.ndarray[np.int32_t, ndim=2] neigh):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        sortCellVerts[int32_t](&cells[0,0], &neigh[0,0], ncells, ndim)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _sortCellVerts_uint32(np.ndarray[np.uint32_t, ndim=2] cells,
                                np.ndarray[np.uint32_t, ndim=2] neigh):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        sortCellVerts[uint32_t](&cells[0,0], &neigh[0,0], ncells, ndim)

@cython.boundscheck(False)
@cython.wraparound(False)
def py_sortCellVerts(cells, neigh):
    r"""Sort the the vertices and neighbors for a single cell such that the 
    vertices are in descending order.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 
        neigh (np.ndarray of int64): (n, m+1) array of neighboring cells 
            for the n cells in a m-dimensional triangulation.
        i (int): Index of cell that should be sorted.

    """
    if len(cells.shape) != 2:
        return
    if cells.shape[0] == 0:
        return
    if cells.dtype == np.int32:
        _sortCellVerts_int32(cells, neigh)
    elif cells.dtype == np.uint32:
        _sortCellVerts_uint32(cells, neigh)
    elif cells.dtype == np.int64:
        _sortCellVerts_int64(cells, neigh)
    elif cells.dtype == np.uint64:
        _sortCellVerts_uint64(cells, neigh)
    else:
        raise TypeError("Type {} not supported.".format(cells.dtype))

def _sortSerializedTess_int32(np.ndarray[np.int32_t, ndim=2] cells, 
                              np.ndarray[np.int32_t, ndim=2] neigh): 
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    sortSerializedTess[int32_t](&cells[0,0], &neigh[0,0], ncells, ndim)
def _sortSerializedTess_uint32(np.ndarray[np.uint32_t, ndim=2] cells, 
                               np.ndarray[np.uint32_t, ndim=2] neigh): 
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    sortSerializedTess[uint32_t](&cells[0,0], &neigh[0,0], ncells, ndim)
def _sortSerializedTess_int64(np.ndarray[np.int64_t, ndim=2] cells, 
                              np.ndarray[np.int64_t, ndim=2] neigh): 
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    sortSerializedTess[int64_t](&cells[0,0], &neigh[0,0], ncells, ndim)
def _sortSerializedTess_uint64(np.ndarray[np.uint64_t, ndim=2] cells, 
                               np.ndarray[np.uint64_t, ndim=2] neigh): 
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    sortSerializedTess[uint64_t](&cells[0,0], &neigh[0,0], ncells, ndim)

def py_sortSerializedTess(cells, neigh):
    r"""Sort serialized triangulation such that the verts for each cell are in 
    descending order, but the cells are sorted in ascending order by the verts.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 
        neigh (np.ndarray of int64): (n, m+1) array of neighboring cells 

    """
    if cells.dtype == np.int32:
        _sortSerializedTess_int32(cells, neigh)
    elif cells.dtype == np.uint32:
        _sortSerializedTess_uint32(cells, neigh)
    elif cells.dtype == np.int64:
        _sortSerializedTess_int64(cells, neigh)
    elif cells.dtype == np.uint64:
        _sortSerializedTess_uint64(cells, neigh)
    else:
        raise TypeError("Type {} not supported.".format(cells.dtype))

def py_quickSort_tess(np.ndarray[np.int64_t, ndim=2] cells,
                      np.ndarray[np.int64_t, ndim=2] neigh,
                      np.ndarray[np.int64_t, ndim=1] idx,
                      int64_t l, int64_t r):
    r"""Sort triangulation between two indices such that vert groups are in 
    ascending order.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 
        neigh (np.ndarray of int64): (n, m+1) array of neighboring cells 
        idx (np.ndarray of int64): (n,) array of indices that will also be 
            sorted in the same order as the cell info.
        l (int): Index of cell to start sort at.
        r (int): Index of cell to stop sort at (inclusive).

    """
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    quickSort_tess[int64_t](&cells[0,0], &neigh[0,0], &idx[0], ndim, l, r)

def py_partition_tess(np.ndarray[np.int64_t, ndim=2] cells,
                      np.ndarray[np.int64_t, ndim=2] neigh,
                      np.ndarray[np.int64_t, ndim=1] idx,
                      int64_t l, int64_t r, int64_t p):
    r"""Partition triangulation cells between two cells by value found at a 
    pivot and return the index of the boundary.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 
        neigh (np.ndarray of int64): (n, m+1) array of neighboring cells 
        idx (np.ndarray of int64): (n,) array of indices that will also be 
            sorted in the same order as the cell info.
        l (int): Index of cell to start partition at.
        r (int): Index of cell to stop partition at (inclusive).
        p (int): Index of cell that should be used as the pivot.

    Returns:
        int: Index of the cell at the boundary between the partitions. Cells 
            with indices less than this index are smaller than the pivot cell 
            and cells with indices greater than index are larger than the pivot 
            cell.

    """
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    return partition_tess[int64_t](&cells[0,0], &neigh[0,0], &idx[0], 
                                   ndim, l, r, p)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _arg_sortCellVerts_int64(np.ndarray[np.int64_t, ndim=2] cells,
                                   np.ndarray[np.uint32_t, ndim=2] idx_verts):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        arg_sortCellVerts[int64_t](&cells[0,0], &idx_verts[0,0], ncells, ndim)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _arg_sortCellVerts_uint64(np.ndarray[np.uint64_t, ndim=2] cells,
                                    np.ndarray[np.uint32_t, ndim=2] idx_verts):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        arg_sortCellVerts[uint64_t](&cells[0,0], &idx_verts[0,0], ncells, ndim)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _arg_sortCellVerts_int32(np.ndarray[np.int32_t, ndim=2] cells,
                                   np.ndarray[np.uint32_t, ndim=2] idx_verts):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        arg_sortCellVerts[int32_t](&cells[0,0], &idx_verts[0,0], ncells, ndim)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _arg_sortCellVerts_uint32(np.ndarray[np.uint32_t, ndim=2] cells,
                                    np.ndarray[np.uint32_t, ndim=2] idx_verts):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    with nogil:
        arg_sortCellVerts[uint32_t](&cells[0,0], &idx_verts[0,0], ncells, ndim)

@cython.boundscheck(False)
@cython.wraparound(False)
def py_arg_sortCellVerts(cells):
    r"""Sort the the vertices and neighbors for a single cell such that the 
    vertices are in descending order.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 
        i (int): Index of cell that should be sorted.

    Returns:
        idx_verts (np.ndarray of uint32): (n, m+1) array of indices to sort 
            vertices in decending order for each cell.

    """
    if len(cells.shape) != 2:
        return
    if cells.shape[0] == 0:
        return
    cdef np.ndarray[np.uint32_t, ndim=2] idx_verts = np.zeros(cells.shape, 'uint32')
    cdef int i
    for i in range(cells.shape[1]):
        idx_verts[:,i] = <np.uint32_t>i
    if cells.dtype == np.int32:
        _arg_sortCellVerts_int32(cells, idx_verts)
    elif cells.dtype == np.uint32:
        _arg_sortCellVerts_uint32(cells, idx_verts)
    elif cells.dtype == np.int64:
        _arg_sortCellVerts_int64(cells, idx_verts)
    elif cells.dtype == np.uint64:
        _arg_sortCellVerts_uint64(cells, idx_verts)
    else:
        raise TypeError("Type {} not supported.".format(cells.dtype))
    return idx_verts

def _arg_sortSerializedTess_int32(np.ndarray[np.int32_t, ndim=2] cells, 
                                  np.ndarray[np.uint32_t, ndim=2] idx_verts,
                                  np.ndarray[np.uint64_t, ndim=1] idx_cells):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    arg_sortSerializedTess[int32_t](&cells[0,0], ncells, ndim, 
                                    &idx_verts[0,0], &idx_cells[0])
def _arg_sortSerializedTess_uint32(np.ndarray[np.uint32_t, ndim=2] cells, 
                                   np.ndarray[np.uint32_t, ndim=2] idx_verts,
                                   np.ndarray[np.uint64_t, ndim=1] idx_cells):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    arg_sortSerializedTess[uint32_t](&cells[0,0], ncells, ndim,
                                     &idx_verts[0,0], &idx_cells[0])
def _arg_sortSerializedTess_int64(np.ndarray[np.int64_t, ndim=2] cells, 
                                  np.ndarray[np.uint32_t, ndim=2] idx_verts,
                                  np.ndarray[np.uint64_t, ndim=1] idx_cells):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    arg_sortSerializedTess[int64_t](&cells[0,0], ncells, ndim,
                                    &idx_verts[0,0], &idx_cells[0])
def _arg_sortSerializedTess_uint64(np.ndarray[np.uint64_t, ndim=2] cells, 
                                   np.ndarray[np.uint32_t, ndim=2] idx_verts,
                                   np.ndarray[np.uint64_t, ndim=1] idx_cells):
    cdef uint64_t ncells = <uint64_t>cells.shape[0]
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    arg_sortSerializedTess[uint64_t](&cells[0,0], ncells, ndim,
                                     &idx_verts[0,0], &idx_cells[0])

def py_arg_sortSerializedTess(cells):
    r"""Sort serialized triangulation such that the verts for each cell are in 
    descending order, but the cells are sorted in ascending order by the verts.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 

    Returns:
        idx_verts (np.ndarray of uint32): (n, m+1) array of indices to sort 
            vertices in decending order for each cell.
        idx_cells (np.ndarray of int64): (n, ) array of indices to sort cells 
            by their sorted vertices.
            
    """
    if len(cells.shape) != 2:
        return
    if cells.shape[0] == 0:
        return
    cdef np.ndarray[np.uint32_t, ndim=2] idx_verts
    cdef np.ndarray[np.uint64_t, ndim=1] idx_cells
    cdef int i
    idx_verts = np.empty(cells.shape, 'uint32')
    for i in range(cells.shape[1]):
        idx_verts[:,i] = <np.uint32_t>i
    idx_cells = np.empty(cells.shape[0], 'uint64')
    for i in range(cells.shape[0]):
        idx_cells[i] = <np.uint64_t>i
    if cells.dtype == np.int32:
        _arg_sortSerializedTess_int32(cells, idx_verts, idx_cells)
    elif cells.dtype == np.uint32:
        _arg_sortSerializedTess_uint32(cells, idx_verts, idx_cells)
    elif cells.dtype == np.int64:
        _arg_sortSerializedTess_int64(cells, idx_verts, idx_cells)
    elif cells.dtype == np.uint64:
        _arg_sortSerializedTess_uint64(cells, idx_verts, idx_cells)
    else:
        raise TypeError("Type {} not supported.".format(cells.dtype))
    return idx_verts, idx_cells

def py_arg_quickSort_tess(np.ndarray[np.int64_t, ndim=2] cells,
                          np.ndarray[np.uint32_t, ndim=2] idx_verts,
                          np.ndarray[np.uint64_t, ndim=1] idx_cells,
                          int64_t l, int64_t r):
    r"""Sort triangulation between two indices such that vert groups are in 
    ascending order.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 
        idx_verts (np.ndarray of uint32): (n, m+1) array of indices to sort 
            vertices in decending order for each cell.
        idx_cells (np.ndarray of int64): (n, ) array of indices to sort cells 
            by their sorted vertices.
        l (int): Index of cell to start sort at.
        r (int): Index of cell to stop sort at (inclusive).

    """
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    arg_quickSort_tess[int64_t](&cells[0,0], &idx_verts[0,0], &idx_cells[0], 
                                ndim, l, r)

def py_arg_partition_tess(np.ndarray[np.int64_t, ndim=2] cells,
                          np.ndarray[np.uint32_t, ndim=2] idx_verts,
                          np.ndarray[np.uint64_t, ndim=1] idx_cells,
                          int64_t l, int64_t r, int64_t p):
    r"""Partition triangulation cells between two cells by value found at a 
    pivot and return the index of the boundary.

    Args:
        cells (np.ndarray of int64): (n, m+1) array of vertex indices  
            for the n cells in a m-dimensional triangulation. 
        idx_verts (np.ndarray of uint32): (n, m+1) array of indices to sort 
            vertices in decending order for each cell.
        idx_cells (np.ndarray of int64): (n, ) array of indices to sort cells 
            by their sorted vertices.
        l (int): Index of cell to start partition at.
        r (int): Index of cell to stop partition at (inclusive).
        p (int): Index of cell that should be used as the pivot.

    Returns:
        int: Index of the cell at the boundary between the partitions. Cells 
            with indices less than this index are smaller than the pivot cell 
            and cells with indices greater than index are larger than the pivot 
            cell.

    """
    cdef uint32_t ndim = <uint32_t>cells.shape[1]
    return arg_partition_tess[int64_t](&cells[0,0], &idx_verts[0,0], &idx_cells[0], 
                                       ndim, l, r, p)
