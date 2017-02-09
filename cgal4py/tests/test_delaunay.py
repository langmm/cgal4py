r"""Tests for generic interface to Delaunay triangulations."""
import numpy as np
from nose.tools import assert_equal
from nose.tools import assert_raises
from cgal4py.delaunay import Delaunay, VoronoiVolumes, tools
from cgal4py.tests.test_delaunay2 import pts as pts2
from cgal4py.tests.test_delaunay2 import left_edge as le2
from cgal4py.tests.test_delaunay2 import right_edge as re2
from cgal4py.tests.test_delaunay3 import pts as pts3
from cgal4py.tests.test_delaunay3 import left_edge as le3
from cgal4py.tests.test_delaunay3 import right_edge as re3
from cgal4py.tests.test_cgal4py import make_points
import copy


pts4, le4, re4 = make_points(10, 4)


def test_Delaunay():
    T2 = Delaunay(pts2)
    assert_equal(T2.num_finite_verts, pts2.shape[0])
    T3 = Delaunay(pts3)
    assert_equal(T3.num_finite_verts, pts3.shape[0])
    T4 = Delaunay(pts4)
    assert_equal(T4.num_finite_verts, pts4.shape[0])
    assert_raises(ValueError, Delaunay, np.zeros((3, 3, 3)))


def test_Delaunay_double():
    T2 = Delaunay(pts2, use_double=True)
    assert_equal(T2.num_finite_verts, pts2.shape[0])
    T3 = Delaunay(pts3, use_double=True)
    assert_equal(T3.num_finite_verts, pts3.shape[0])
    T4 = Delaunay(pts4)
    assert_equal(T4.num_finite_verts, pts4.shape[0])
    assert_raises(ValueError, Delaunay, np.zeros((3, 3, 3)), True)


def test_Delaunay_periodic():
    T2 = Delaunay(pts2, periodic=True, left_edge=le2, right_edge=re2)
    assert_equal(T2.num_verts, pts2.shape[0])
    T3 = Delaunay(pts3, periodic=True, left_edge=le3, right_edge=re3)
    assert_equal(T3.num_verts, pts3.shape[0])
    assert_raises(NotImplementedError, Delaunay, np.zeros((3, 4)),
                  False, True, le4, re4)
    assert_raises(ValueError, Delaunay, np.zeros((3, 3, 3)))


def test_Delaunay_both():
    T2 = Delaunay(pts2, use_double=True, periodic=True,
                  left_edge=le2, right_edge=re2)
    assert_equal(T2.num_verts, pts2.shape[0])
    T3 = Delaunay(pts3, use_double=True, periodic=True,
                  left_edge=le3, right_edge=re3)
    assert_equal(T3.num_verts, pts3.shape[0])
    assert_raises(NotImplementedError, Delaunay, np.zeros((3, 4)),
                  True, True, le4, re4)
    assert_raises(ValueError, Delaunay, np.zeros((3, 3, 3)))


def test_VoronoiVolumes():
    T2 = VoronoiVolumes(pts2)
    assert_equal(T2.shape[0], pts2.shape[0])
    T3 = VoronoiVolumes(pts3)
    assert_equal(T3.shape[0], pts3.shape[0])
    T4 = VoronoiVolumes(pts4)
    assert_equal(T4.shape[0], pts4.shape[0])


def test_VoronoiVolumes_double():
    T2 = VoronoiVolumes(pts2, use_double=True)
    assert_equal(T2.shape[0], pts2.shape[0])
    T3 = VoronoiVolumes(pts3, use_double=True)
    assert_equal(T3.shape[0], pts3.shape[0])
    T4 = VoronoiVolumes(pts4, use_double=True)
    assert_equal(T4.shape[0], pts4.shape[0])


def test_VoronoiVolumes_periodic():
    T2 = VoronoiVolumes(pts2, periodic=True, left_edge=le2, right_edge=re2)
    assert_equal(T2.shape[0], pts2.shape[0])
    T3 = VoronoiVolumes(pts3, periodic=True, left_edge=le3, right_edge=re3)
    assert_equal(T3.shape[0], pts3.shape[0])
    assert_raises(NotImplementedError, VoronoiVolumes,
                  np.zeros((3, 4)), False, True, le4, re4)


def test_VoronoiVolumes_both():
    T2 = VoronoiVolumes(pts2, use_double=True, periodic=True,
                        left_edge=le2, right_edge=re2)
    assert_equal(T2.shape[0], pts2.shape[0])
    T3 = VoronoiVolumes(pts3, use_double=True, periodic=True,
                        left_edge=le3, right_edge=re3)
    assert_equal(T3.shape[0], pts3.shape[0])
    assert_raises(NotImplementedError, VoronoiVolumes,
                  np.zeros((3, 4)), True, True, le4, re4)


# Tools
def test_intersect_sph_box():
    for ndim in [2, 3]:
        c = np.zeros(ndim, 'float64')
        r = np.float64(1)
        # Sphere inside box
        le = -1*np.ones(ndim, 'float64')
        re = 1*np.ones(ndim, 'float64')
        print(c.shape)
        print(le.shape)
        print(c.shape == le.shape)
        assert(tools.py_intersect_sph_box(c, r, le, re) == True)
        # Box inside sphere
        le = -np.sqrt(2.0)*np.ones(ndim, 'float64')
        re = np.sqrt(2.0)*np.ones(ndim, 'float64')
        assert(tools.py_intersect_sph_box(c, r, le, re) == True)
        # Box half inside sphere
        le = np.zeros(ndim, 'float64')
        re = 2*np.ones(ndim, 'float64')
        assert(tools.py_intersect_sph_box(c, r, le, re) == True)
        # Box touching sphere
        le = np.ones(ndim, 'float64')
        re = 2*np.ones(ndim, 'float64')
        assert(tools.py_intersect_sph_box(c, r, le, re) == True)
        # Box outside sphere
        le = 2*np.ones(ndim, 'float64')
        re = 3*np.ones(ndim, 'float64')
        assert(tools.py_intersect_sph_box(c, r, le, re) == False)


def test_arg_tLT():
    cells = np.array([[2, 1, 0],
                      [3, 1, 0],
                      [2, 2, 0],
                      [2, 1, 1]], 'int64')
    idx_verts = np.empty(cells.shape, 'uint32')
    for i in range(cells.shape[1]):
        idx_verts[:, i] = i
    assert(tools.py_arg_tLT(cells, idx_verts, 0, 0) == False)
    assert(tools.py_arg_tLT(cells, idx_verts, 0, 1) == True)
    assert(tools.py_arg_tLT(cells, idx_verts, 0, 2) == True)
    assert(tools.py_arg_tLT(cells, idx_verts, 0, 3) == True)
    assert(tools.py_arg_tLT(cells, idx_verts, 1, 0) == False)
    assert(tools.py_arg_tLT(cells, idx_verts, 2, 0) == False)
    assert(tools.py_arg_tLT(cells, idx_verts, 3, 0) == False)


def test_tEQ():
    cells = np.array([[2, 1, 0],
                      [3, 1, 0],
                      [2, 2, 0],
                      [2, 1, 1]], 'int64')
    assert(tools.py_tEQ(cells, 0, 0) == True)
    assert(tools.py_tEQ(cells, 0, 1) == False)
    assert(tools.py_tEQ(cells, 0, 2) == False)
    assert(tools.py_tEQ(cells, 0, 3) == False)
    assert(tools.py_tEQ(cells, 1, 0) == False)
    assert(tools.py_tEQ(cells, 2, 0) == False)
    assert(tools.py_tEQ(cells, 3, 0) == False)


def test_tGT():
    cells = np.array([[2, 1, 0],
                      [3, 1, 0],
                      [2, 2, 0],
                      [2, 1, 1]], 'int64')
    assert(tools.py_tGT(cells, 0, 0) == False)
    assert(tools.py_tGT(cells, 0, 1) == False)
    assert(tools.py_tGT(cells, 0, 2) == False)
    assert(tools.py_tGT(cells, 0, 3) == False)
    assert(tools.py_tGT(cells, 1, 0) == True)
    assert(tools.py_tGT(cells, 2, 0) == True)
    assert(tools.py_tGT(cells, 3, 0) == True)


def test_tLT():
    cells = np.array([[2, 1, 0],
                      [3, 1, 0],
                      [2, 2, 0],
                      [2, 1, 1]], 'int64')
    assert(tools.py_tLT(cells, 0, 0) == False)
    assert(tools.py_tLT(cells, 0, 1) == True)
    assert(tools.py_tLT(cells, 0, 2) == True)
    assert(tools.py_tLT(cells, 0, 3) == True)
    assert(tools.py_tLT(cells, 1, 0) == False)
    assert(tools.py_tLT(cells, 2, 0) == False)
    assert(tools.py_tLT(cells, 3, 0) == False)


def test_sortCellVerts():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'uint32') - 1
    neigh = np.zeros((npts, ndim), 'uint32') - 1
    for i in range(npts):
        for j in range(ndim):
            x = np.random.randint(0, npts)
            while x in cells[i, :j]:
                x = np.random.randint(0, npts)
            cells[i, j] = x
            neigh[i, j] = np.random.randint(0, npts)
    cells0 = copy.copy(cells)
    neigh0 = copy.copy(neigh)
    tools.py_sortCellVerts(cells, neigh)
    for i in range(npts):
        assert(np.all(cells[i, :] == np.sort(cells0[i, :])[::-1]))
        assert(np.all(neigh[i, :] ==
                      neigh0[i, np.argsort(cells0[i, :])[::-1]]))


def test_partition_tess():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'int64')
    neigh = np.zeros((npts, ndim), 'int64')
    idx = np.arange(npts).astype('int64')
    for i in range(npts):
        for j in range(ndim):
            cells[i, j] = np.random.randint(0, npts)
            neigh[i, j] = np.random.randint(0, npts)
    cells0 = copy.copy(cells)
    pivot = copy.copy(cells[0, :])
    x = tools.py_partition_tess(cells, neigh, idx, 0, npts-1, 0)
    for i1, i0 in enumerate(idx):
        for d in range(ndim):
            assert(cells0[i0, d] == cells[i1, d])
    for i in range(x):
        for d in range(ndim):
            if cells[i, d] != pivot[d]:
                break
        assert(cells[i, d] < pivot[d])
    for i in range(x, npts):
        for d in range(ndim):
            if cells[i, d] != pivot[d]:
                break
        assert(cells[i, d] >= pivot[d])


def test_quickSort_tess():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'int64')
    neigh = np.zeros((npts, ndim), 'int64')
    idx = np.arange(npts).astype('int64')
    for i in range(npts):
        for j in range(ndim):
            cells[i, j] = np.random.randint(0, npts)
            neigh[i, j] = np.random.randint(0, npts)
    cells0 = copy.copy(cells)
    tools.py_quickSort_tess(cells, neigh, idx, 0, npts-1)
    for i1, i0 in enumerate(idx):
        for d in range(ndim):
            assert(cells0[i0, d] == cells[i1, d])
    for i in range(1, npts):
        for d in range(ndim):
            if cells[i, d] != cells[i-1, d]:
                break
        # print(d, cells[i, :], cells[i-1, :], tools.py_tGT(cells, i, i-1))
        assert(cells[i, d] >= cells[i-1, d])


def test_sortSerializedTess():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'int64')
    neigh = np.zeros((npts, ndim), 'int64')
    for i in range(npts):
        for j in range(ndim):
            cells[i, j] = np.random.randint(0, npts)
            neigh[i, j] = np.random.randint(0, npts)
    tools.py_sortSerializedTess(cells, neigh)
    for i in range(1, npts):
        assert(np.all(cells[i, :] == np.sort(cells[i, :])[::-1]))
        for d in range(ndim):
            if cells[i, d] != cells[i-1, d]:
                break
        assert(cells[i, d] >= cells[i-1, d])


# argsort
def test_arg_sortCellVerts():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'uint32') - 1
    for i in range(npts):
        for j in range(ndim):
            x = np.random.randint(0, npts)
            while x in cells[i, :j]:
                x = np.random.randint(0, npts)
            cells[i, j] = x
    cells0 = copy.copy(cells)
    idx_verts = tools.py_arg_sortCellVerts(cells)
    for i in range(npts):
        assert(np.all(cells[i, idx_verts[i, :]] ==
                      np.sort(cells0[i, :])[::-1]))


def test_arg_partition_tess():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'int64')
    idx_verts = np.empty((npts, ndim), 'uint32')
    for i in range(ndim):
        idx_verts[:, i] = i
    idx_cells = np.arange(npts).astype('uint64')
    for i in range(npts):
        for j in range(ndim):
            cells[i, j] = np.random.randint(0, npts)
    pivot = copy.copy(cells[0, :])
    x = tools.py_arg_partition_tess(cells, idx_verts, idx_cells, 0, npts-1, 0)
    for i in range(x):
        for d in range(ndim):
            if cells[idx_cells[i], d] != pivot[d]:
                break
        assert(cells[idx_cells[i], d] < pivot[d])
    for i in range(x, npts):
        for d in range(ndim):
            if cells[idx_cells[i], d] != pivot[d]:
                break
        assert(cells[idx_cells[i], d] >= pivot[d])


def test_arg_quickSort_tess():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'int64')
    idx_verts = np.empty((npts, ndim), 'uint32')
    for i in range(ndim):
        idx_verts[:, i] = i
    idx_cells = np.arange(npts).astype('uint64')
    for i in range(npts):
        for j in range(ndim):
            cells[i, j] = np.random.randint(0, npts)
    tools.py_arg_quickSort_tess(cells, idx_verts, idx_cells, 0, npts-1)
    for i in range(1, npts):
        for d in range(ndim):
            if cells[idx_cells[i], d] != cells[idx_cells[i-1], d]:
                break
        assert(cells[idx_cells[i], d] >= cells[idx_cells[i-1], d])


def test_arg_sortSerializedTess():
    npts = 20
    ndim = 3
    cells = np.zeros((npts, ndim), 'int64')
    for i in range(npts):
        for j in range(ndim):
            cells[i, j] = np.random.randint(0, npts)
    idx_verts, idx_cells = tools.py_arg_sortSerializedTess(cells)
    for i in range(1, npts):
        assert(np.all(cells[i, idx_verts[i, :]] == np.sort(cells[i, :])[::-1]))
        for d in range(ndim):
            if (cells[idx_cells[i], idx_verts[i, d]] !=
                    cells[idx_cells[i-1], idx_verts[i, d]]):
                break
        i_old = idx_cells[i-1]
        i_new = idx_cells[i]
        assert(cells[i_new, idx_verts[i_new, d]] >=
               cells[i_old, idx_verts[i_old, d]])
