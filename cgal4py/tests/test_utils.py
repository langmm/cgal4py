import numpy as np

from .. import utils

def test_max_pts():
    pts = np.arange(5*3).reshape((5,3)).astype('float64')
    out = utils.py_max_pts(pts)
    assert(np.allclose(out, np.max(pts, axis=0)))

def test_min_pts():
    pts = np.arange(5*3).reshape((5,3)).astype('float64')
    out = utils.py_min_pts(pts)
    assert(np.allclose(out, np.min(pts, axis=0)))

def test_quickSort():
    d = 1
    np.random.seed(10)
    # Even number
    N = 10
    pts = np.random.rand(N,2).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:,d])))
    pts = np.random.rand(N,3).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:,d])))
    # Odd number
    N = 11
    pts = np.random.rand(N,2).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:,d])))
    pts = np.random.rand(N,3).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:,d])))


