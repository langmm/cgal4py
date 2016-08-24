import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
from cgal4py import triangulate
from test_delaunay2 import pts as pts2
from test_delaunay3 import pts as pts3

def test_Delaunay():
    T2 = triangulate(pts2)
    T3 = triangulate(pts3)
    assert_raises(NotImplementedError, triangulate, pts2, periodic=True)
    assert_raises(ValueError, triangulate, np.zeros((3,3,3)))
    assert_raises(ValueError, triangulate, pts2, left_edge=np.zeros(3))
    assert_raises(ValueError, triangulate, pts2, right_edge=np.zeros(3))
    assert_raises(ValueError, triangulate, pts2, left_edge=np.zeros((2,2,2)))
    assert_raises(ValueError, triangulate, pts2, right_edge=np.zeros((2,2,2)))
    assert_raises(NotImplementedError, triangulate, pts2, limit_mem=True)
