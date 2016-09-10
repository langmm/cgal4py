import numpy as np
from nose import with_setup
from nose.tools import assert_equal, assert_raises, nottest
from cgal4py import triangulate, domain_decomp
from test_delaunay2 import pts as pts2
from test_delaunay2 import left_edge as left_edge2
from test_delaunay2 import right_edge as right_edge2
from test_delaunay3 import pts as pts3
from test_delaunay3 import left_edge as left_edge3
from test_delaunay3 import right_edge as right_edge3

@nottest
def make_test(npts, ndim, distrib='uniform', periodic=False, leafsize=None):
    # Points
    if npts <= 0:
        if ndim == 2:
            pts = pts2
            left_edge = left_edge2
            right_edge = right_edge2
        elif ndim == 3:
            pts = pts3
            left_edge = left_edge3
            right_edge = right_edge3
        else:
            raise ValueError("Invalid 'ndim': {}".format(ndim))
        npts = pts.shape[0]
    else:
        LE = 0.0; RE = 1.0
        left_edge = LE*np.ones(ndim, 'float64')
        right_edge = RE*np.ones(ndim, 'float64')
        if distrib == 'uniform':
            pts = np.random.uniform(low=LE, high=RE, size=(npts, ndim))
        elif distrib in ('gaussian','normal'):
            pts = np.random.normal(loc=(LE+RE)/2.0, scale=(RE-LE)/4.0, size=(npts, ndim))
            np.clip(pts, LE, RE)
        elif distrib in (2, '2'):
            pts = pts2
        elif distrib in (3, '3'):
            pts = pts3
        else:
            raise ValueError("Invalid 'distrib': {}".format(distrib))
    # Tree
    if leafsize is None:
        leafsize = npts/2 + 2
    tree = domain_decomp.tree("kdtree", pts, left_edge, right_edge,
                              periodic=periodic, leafsize=leafsize)
    return pts, tree

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
