import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
from cgal4py import Triangulate, domain_decomp, parallel
from test_delaunay2 import pts as pts2
from test_delaunay2 import left_edge as left_edge2
from test_delaunay2 import right_edge as right_edge2
from test_delaunay3 import pts as pts3
from test_delaunay3 import left_edge as left_edge3
from test_delaunay3 import right_edge as right_edge3

leaves2 = domain_decomp.leaves("kdtree", pts2, left_edge2, right_edge2,
                               periodic=False, leafsize=pts2.shape[0]/2 + 2)
leaves3 = domain_decomp.leaves("kdtree", pts3, left_edge3, right_edge3,
                               periodic=False, leafsize=pts3.shape[0]/2 + 2)
leaves2_periodic = domain_decomp.leaves("kdtree", pts2, left_edge2, right_edge2,
                                        periodic=True, leafsize=pts2.shape[0]/2 + 2)
leaves3_periodic = domain_decomp.leaves("kdtree", pts3, left_edge3, right_edge3,
                                        periodic=True, leafsize=pts3.shape[0]/2 + 2)
assert(len(leaves2) == 2)
assert(len(leaves3) == 2)

# TODO: Testing of parallel code with coverage

def test_ParallelLeaf():
    out2 = []
    for i,leaf in enumerate(leaves2):
        parallel.parallelize_leaf(leaf, pts2)
        assert(leaf.id == i)
        out2.append(leaf.outgoing_points())
    leaves2[0].incoming_points(1, out2[1][0], pts2[out2[1][0],:])
    leaves2[0].incoming_points(0, out2[0][0], pts2[out2[0][0],:])
    out3 = []
    for i,leaf in enumerate(leaves3):
        parallel.parallelize_leaf(leaf, pts3)
        assert(leaf.id == i)
        out3.append(leaf.outgoing_points())
    leaves3[0].incoming_points(1, out3[1][0], pts3[out3[1][0],:])
    leaves3[0].incoming_points(0, out3[0][0], pts3[out3[0][0],:])

def test_ParallelLeaf_periodic():
    out2 = []
    for i,leaf in enumerate(leaves2_periodic):
        parallel.parallelize_leaf(leaf, pts2)
        assert(leaf.id == i)
        out2.append(leaf.outgoing_points())
    leaves2_periodic[0].incoming_points(1, out2[1][0], pts2[out2[1][0],:])
    leaves2_periodic[0].incoming_points(0, out2[0][0], pts2[out2[0][0],:])
    out3 = []
    for i,leaf in enumerate(leaves3_periodic):
        parallel.parallelize_leaf(leaf, pts3)
        assert(leaf.id == i)
        out3.append(leaf.outgoing_points())
    leaves3_periodic[0].incoming_points(1, out3[1][0], pts3[out3[1][0],:])
    leaves3_periodic[0].incoming_points(0, out3[0][0], pts3[out3[0][0],:])

def test_parallel_Triangulate():
    T2 = Triangulate(pts2, dd_kwargs={'leafsize':2}, nproc=5)
    T3 = Triangulate(pts3, dd_kwargs={'leafsize':2}, nproc=5)
    T2 = Triangulate(pts2, dd_kwargs={'leafsize':2}, nproc=5, periodic=True)
    T3 = Triangulate(pts3, dd_kwargs={'leafsize':2}, nproc=5, periodic=True)
