import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
from cgal4py import Triangulate
from test_delaunay2 import pts as pts2
from test_delaunay3 import pts as pts3

def test_Delaunay():
    T2 = Triangulate(pts2)
    T3 = Triangulate(pts3)
    assert_raises(NotImplementedError, Triangulate, pts2, periodic=True)
    assert_raises(ValueError, Triangulate, np.zeros((3,3,3)))
    assert_raises(ValueError, Triangulate, pts2, left_edge=np.zeros(3))
    assert_raises(ValueError, Triangulate, pts2, right_edge=np.zeros(3))
    assert_raises(ValueError, Triangulate, pts2, left_edge=np.zeros((2,2,2)))
    assert_raises(ValueError, Triangulate, pts2, right_edge=np.zeros((2,2,2)))
