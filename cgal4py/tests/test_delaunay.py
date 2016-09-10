import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
from cgal4py.delaunay import Delaunay, tools
from test_delaunay2 import pts as pts2
from test_delaunay3 import pts as pts3
import copy

def test_Delaunay():
    T2 = Delaunay(pts2)
    assert_equal(T2.num_finite_verts, pts2.shape[0])
    T3 = Delaunay(pts3)
    assert_equal(T3.num_finite_verts, pts3.shape[0])
    assert_raises(NotImplementedError, Delaunay, np.zeros((3,4)))
    assert_raises(ValueError, Delaunay, np.zeros((3,3,3)))

def test_Delaunay_double():
    T2 = Delaunay(pts2, use_double=True)
    assert_equal(T2.num_finite_verts, pts2.shape[0])
    T3 = Delaunay(pts3, use_double=True)
    assert_equal(T3.num_finite_verts, pts3.shape[0])
    assert_raises(NotImplementedError, Delaunay, np.zeros((3,4)), True)
    assert_raises(ValueError, Delaunay, np.zeros((3,3,3)), True)

# Tools
def test_tEQ():
    cells = np.array([[2,1,0],
                      [3,1,0],
                      [2,2,0],
                      [2,1,1]], 'int64')
    assert(tools.py_tEQ(cells, 0, 0) == True)
    assert(tools.py_tEQ(cells, 0, 1) == False)
    assert(tools.py_tEQ(cells, 0, 2) == False)
    assert(tools.py_tEQ(cells, 0, 3) == False)
    assert(tools.py_tEQ(cells, 1, 0) == False)
    assert(tools.py_tEQ(cells, 2, 0) == False)
    assert(tools.py_tEQ(cells, 3, 0) == False)

def test_tGT():
    cells = np.array([[2,1,0],
                      [3,1,0],
                      [2,2,0],
                      [2,1,1]], 'int64')
    assert(tools.py_tGT(cells, 0, 0) == False)
    assert(tools.py_tGT(cells, 0, 1) == False)
    assert(tools.py_tGT(cells, 0, 2) == False)
    assert(tools.py_tGT(cells, 0, 3) == False)
    assert(tools.py_tGT(cells, 1, 0) == True)
    assert(tools.py_tGT(cells, 2, 0) == True)
    assert(tools.py_tGT(cells, 3, 0) == True)

def test_tLT():
    cells = np.array([[2,1,0],
                      [3,1,0],
                      [2,2,0],
                      [2,1,1]], 'int64')
    assert(tools.py_tLT(cells, 0, 0) == False)
    assert(tools.py_tLT(cells, 0, 1) == True)
    assert(tools.py_tLT(cells, 0, 2) == True)
    assert(tools.py_tLT(cells, 0, 3) == True)
    assert(tools.py_tLT(cells, 1, 0) == False)
    assert(tools.py_tLT(cells, 2, 0) == False)
    assert(tools.py_tLT(cells, 3, 0) == False)

def test_sortCellVerts():
    npts = 20; ndim = 3;
    cells = np.zeros((npts, ndim), 'uint32') - 1
    neigh = np.zeros((npts, ndim), 'uint32') - 1
    for i in range(npts):
        for j in range(ndim):
            x = np.random.randint(0,npts)
            while x in cells[i,:j]:
                x = np.random.randint(0,npts)
            cells[i,j] = x
            neigh[i,j] = np.random.randint(0,npts)
    cells0 = copy.copy(cells)
    neigh0 = copy.copy(neigh)
    tools.py_sortCellVerts(cells, neigh)
    for i in range(npts):
        assert(np.all(cells[i,:] == np.sort(cells0[i,:])[::-1]))
        assert(np.all(neigh[i,:] == neigh0[i,np.argsort(cells0[i,:])[::-1]]))

def test_partition_tess():
    npts = 20; ndim = 3;
    cells = np.zeros((npts, ndim), 'int64')
    neigh = np.zeros((npts, ndim), 'int64')
    idx = np.arange(npts).astype('int64')
    for i in range(npts):
        for j in range(ndim):
            cells[i,j] = np.random.randint(0,npts)
            neigh[i,j] = np.random.randint(0,npts)
    cells0 = copy.copy(cells)
    neigh0 = copy.copy(neigh)
    pivot = copy.copy(cells[0,:])
    x = tools.py_partition_tess(cells, neigh, idx, 0, npts-1, 0)
    for i1, i0 in enumerate(idx):
        for d in range(ndim):
            assert(cells0[i0,d] == cells[i1, d])
    for i in range(x):
        for d in range(ndim):
            if cells[i,d] != pivot[d]:
                break
        assert(cells[i,d] < pivot[d])
    for i in range(x, npts):
        for d in range(ndim):
            if cells[i,d] != pivot[d]:
                break
        assert(cells[i,d] >= pivot[d])

def test_quickSort_tess():
    npts = 20; ndim = 3;
    cells = np.zeros((npts, ndim), 'int64')
    neigh = np.zeros((npts, ndim), 'int64')
    idx = np.arange(npts).astype('int64')
    for i in range(npts):
        for j in range(ndim):
            cells[i,j] = np.random.randint(0,npts)
            neigh[i,j] = np.random.randint(0,npts)
    cells0 = copy.copy(cells)
    neigh0 = copy.copy(neigh)
    tools.py_quickSort_tess(cells, neigh, idx, 0, npts-1)
    for i1, i0 in enumerate(idx):
        for d in range(ndim):
            assert(cells0[i0, d] == cells[i1, d])
    for i in range(1,npts):
        for d in range(ndim):
            if cells[i,d] != cells[i-1,d]:
                break
        # print d, cells[i,:], cells[i-1,:], tools.py_tGT(cells, i, i-1)
        assert(cells[i,d] >= cells[i-1,d])

def test_sortSerializedTess():
    npts = 20; ndim = 3;
    cells = np.zeros((npts, ndim), 'int64')
    neigh = np.zeros((npts, ndim), 'int64')
    for i in range(npts):
        for j in range(ndim):
            cells[i,j] = np.random.randint(0,npts)
            neigh[i,j] = np.random.randint(0,npts)
    cells0 = copy.copy(cells)
    neigh0 = copy.copy(neigh)
    tools.py_sortSerializedTess(cells, neigh)
    for i in range(1,npts):
        assert(np.all(cells[i,:] == np.sort(cells[i,:])[::-1]))
        for d in range(ndim):
            if cells[i,d] != cells[i-1,d]:
                break
        assert(cells[i,d] >= cells[i-1,d])
