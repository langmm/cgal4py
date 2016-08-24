import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
import multiprocessing as mp
from cgal4py import domain_decomp, parallel
from test_delaunay2 import pts as pts2
from test_delaunay2 import left_edge as left_edge2
from test_delaunay2 import right_edge as right_edge2
from test_delaunay3 import pts as pts3
from test_delaunay3 import left_edge as left_edge3
from test_delaunay3 import right_edge as right_edge3
import time

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

def test_parallelize_leaf():
    leaf = leaves2[0]
    parallel.parallelize_leaf(leaf, pts2) # Regular leaf
    parallel.parallelize_leaf(leaf, pts2) # Parallel leaf already
    leaf = None
    assert_raises(ValueError, parallel.parallelize_leaf, leaf, pts2)

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

def test_ParallelDelaunay():
    T2 = parallel.ParallelDelaunay(pts2, leaves2, 2)
    T3 = parallel.ParallelDelaunay(pts3, leaves3, 2)

def test_ParallelDelaunay_periodic():
    T2 = parallel.ParallelDelaunay(pts2, leaves2_periodic, 2)
    T3 = parallel.ParallelDelaunay(pts3, leaves3_periodic, 2)

def test_DelaunayProcess2():
    pts = pts2
    leaves = leaves2
    nproc = 2 # len(leaves)
    queues = [mp.Queue() for _ in xrange(nproc)]
    # Split leaves 
    task2leaves = [[] for _ in xrange(nproc)]
    for leaf in leaves:
        task = leaf.id % nproc
        task2leaves[task].append(leaf)
    # Create processes & tessellate
    processes = []
    for i in xrange(nproc):
        P = parallel.DelaunayProcess(task2leaves[i], pts, queues, i)
        processes.append(P)
    # Split
    P1, P2 = processes[0], processes[1]
    # Do partial run on 1
    P1.tessellate_leaves()
    P1.outgoing_points()
    # Full run on 2
    P2.run()
    # Finish on 1
    i,j,arr = queues[0].get()
    queues[0].put((i,j,np.array([])))
    P1.incoming_points()
    P1.finalize_process()
    time.sleep(0.01)

    # # Tessellate
    # for P in processes:
    #     P.tessellate_leaves()
    # # Outgoing
    # for P in processes:
    #     P.outgoing_points()
    # i,j,arr = queues[0].get()
    # queues[0].put((i,j,np.array([])))
    # # Incoming
    # for P in processes:
    #     P.incoming_points()
    # # Finalize
    # for P in processes:
    #     time.sleep(0.01)
    #     P.finalize_process()
    #     time.sleep(0.01)

