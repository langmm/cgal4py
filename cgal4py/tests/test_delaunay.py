import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
from cgal4py.delaunay import Delaunay
from test_delaunay2 import pts as pts2
from test_delaunay3 import pts as pts3

def test_Delaunay():
    T2 = Delaunay(pts2)
    assert_equal(T2.num_finite_verts, pts2.shape[0])
    T3 = Delaunay(pts3)
    assert_equal(T3.num_finite_verts, pts3.shape[0])
    assert_raises(NotImplementedError, Delaunay, np.zeros((3,4)))
    assert_raises(ValueError, Delaunay, np.zeros((3,3,3)))

