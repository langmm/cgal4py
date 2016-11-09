from .. import utils
import numpy as np


def test_max_pts():
    pts = np.arange(5*3).reshape((5, 3)).astype('float64')
    out = utils.py_max_pts(pts)
    assert(np.allclose(out, np.max(pts, axis=0)))


def test_min_pts():
    pts = np.arange(5*3).reshape((5, 3)).astype('float64')
    out = utils.py_min_pts(pts)
    assert(np.allclose(out, np.min(pts, axis=0)))


def test_quickSort():
    d = 1
    np.random.seed(10)
    # Even number
    N = 10
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))
    # Odd number
    N = 11
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_quickSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))


def test_insertSort():
    d = 1
    np.random.seed(10)
    # Even number
    N = 10
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_insertSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_insertSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))
    # Odd number
    N = 11
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_insertSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_insertSort(pts, d)
    assert(np.allclose(idx, np.argsort(pts[:, d])))


def test_pivot():
    d = 1
    np.random.seed(10)
    # Even number
    N = 10
    pts = np.random.rand(N, 2).astype('float64')
    q, idx = utils.py_pivot(pts, d)
    assert((pts[idx[:q], d] <= pts[idx[q], d]).all())
    pts = np.random.rand(N, 3).astype('float64')
    q, idx = utils.py_pivot(pts, d)
    assert((pts[idx[:q], d] <= pts[idx[q], d]).all())
    # Odd number
    N = 11
    pts = np.random.rand(N, 2).astype('float64')
    q, idx = utils.py_pivot(pts, d)
    assert((pts[idx[:q], d] <= pts[idx[q], d]).all())
    pts = np.random.rand(N, 3).astype('float64')
    q, idx = utils.py_pivot(pts, d)
    assert((pts[idx[:q], d] <= pts[idx[q], d]).all())


def test_partition():
    d = 1
    p = 0
    np.random.seed(10)
    # Even number
    N = 10
    pts = np.random.rand(N, 2).astype('float64')
    q, idx = utils.py_partition(pts, d, p)
    assert((pts[idx[:q], d] <= pts[p, d]).all())
    assert((pts[idx[q:], d] > pts[p, d]).all())
    pts = np.random.rand(N, 3).astype('float64')
    q, idx = utils.py_partition(pts, d, p)
    assert((pts[idx[:q], d] <= pts[p, d]).all())
    assert((pts[idx[q:], d] > pts[p, d]).all())
    # Odd number
    N = 11
    pts = np.random.rand(N, 2).astype('float64')
    q, idx = utils.py_partition(pts, d, p)
    assert((pts[idx[:q], d] <= pts[p, d]).all())
    assert((pts[idx[q:], d] > pts[p, d]).all())
    pts = np.random.rand(N, 3).astype('float64')
    q, idx = utils.py_partition(pts, d, p)
    assert((pts[idx[:q], d] <= pts[p, d]).all())
    assert((pts[idx[q:], d] > pts[p, d]).all())


def test_select():
    d = 1
    np.random.seed(10)
    # Even number
    N = 10
    p = q = N/2
    p -= 1
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
    # Odd number
    N = 11
    p = q = N/2
    q += 1
    pts = np.random.rand(N, 2).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
    pts = np.random.rand(N, 3).astype('float64')
    idx = utils.py_select(pts, d, p)
    assert((pts[idx[:q], d] <= np.median(pts[:, d])).all())
    assert((pts[idx[q:], d] > np.median(pts[:, d])).all())
