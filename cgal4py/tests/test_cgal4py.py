r"""Tests for package level methods."""
import numpy as np
from nose.tools import assert_raises, nottest
from cgal4py import triangulate, voronoi_volumes, domain_decomp
from test_delaunay2 import pts as pts2
from test_delaunay2 import left_edge as left_edge2
from test_delaunay2 import right_edge as right_edge2
from test_delaunay3 import pts as pts3
from test_delaunay3 import left_edge as left_edge3
from test_delaunay3 import right_edge as right_edge3


@nottest
def make_points(npts, ndim, distrib='uniform'):
    npts = int(npts)
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
        LE = 0.0
        RE = 1.0
        left_edge = LE*np.ones(ndim, 'float64')
        right_edge = RE*np.ones(ndim, 'float64')
        if distrib == 'uniform':
            pts = np.random.uniform(low=LE, high=RE, size=(npts, ndim))
        elif distrib in ('gaussian', 'normal'):
            pts = np.random.normal(loc=(LE+RE)/2.0, scale=(RE-LE)/4.0,
                                   size=(npts, ndim))
            np.clip(pts, LE, RE)
        elif distrib in (2, '2'):
            pts = pts2
        elif distrib in (3, '3'):
            pts = pts3
        else:
            raise ValueError("Invalid 'distrib': {}".format(distrib))
    return pts, left_edge, right_edge


@nottest
def make_test(npts, ndim, distrib='uniform', periodic=False,
              leafsize=None, nleaves=0):
    # Points
    pts, left_edge, right_edge = make_points(npts, ndim, distrib=distrib)
    npts = pts.shape[0]
    # Tree
    if leafsize is None:
        leafsize = npts/2 + 2
    tree = domain_decomp.tree("kdtree", pts, left_edge, right_edge,
                              periodic=periodic, leafsize=leafsize,
                              nleaves=nleaves)
    return pts, tree


def test_triangulate():
    T2 = triangulate(pts2)
    T3 = triangulate(pts3)
    T2 = triangulate(pts2, periodic=True,
                     left_edge=left_edge2, right_edge=right_edge2)
    T3 = triangulate(pts3, periodic=True,
                     left_edge=left_edge3, right_edge=right_edge3)
    del(T2, T3)
    assert_raises(ValueError, triangulate, np.zeros((3, 3, 3)))
    assert_raises(ValueError, triangulate, pts2, left_edge=np.zeros(3))
    assert_raises(ValueError, triangulate, pts2, right_edge=np.zeros(3))
    assert_raises(ValueError, triangulate, pts2,
                  left_edge=np.zeros((2, 2, 2)))
    assert_raises(ValueError, triangulate, pts2,
                  right_edge=np.zeros((2, 2, 2)))
    assert_raises(NotImplementedError, triangulate, pts2, limit_mem=True)


def test_voronoi_volumes():
    v2 = voronoi_volumes(pts2)
    v3 = voronoi_volumes(pts3)
    v2 = voronoi_volumes(pts2, periodic=True,
                         left_edge=left_edge2, right_edge=right_edge2)
    v3 = voronoi_volumes(pts3, periodic=True,
                         left_edge=left_edge3, right_edge=right_edge3)
    del(v2, v3)
    assert_raises(ValueError, voronoi_volumes, np.zeros((3, 3, 3)))
    assert_raises(ValueError, voronoi_volumes, pts2, left_edge=np.zeros(3))
    assert_raises(ValueError, voronoi_volumes, pts2, right_edge=np.zeros(3))
    assert_raises(ValueError, voronoi_volumes, pts2,
                  left_edge=np.zeros((2, 2, 2)))
    assert_raises(ValueError, voronoi_volumes, pts2,
                  right_edge=np.zeros((2, 2, 2)))
    assert_raises(NotImplementedError, voronoi_volumes, pts2, limit_mem=True)
