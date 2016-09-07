import numpy as np
from nose import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises
import multiprocessing as mp
from cgal4py import domain_decomp, parallel, delaunay
from test_delaunay2 import pts as pts2
from test_delaunay2 import left_edge as left_edge2
from test_delaunay2 import right_edge as right_edge2
from test_delaunay3 import pts as pts3
from test_delaunay3 import left_edge as left_edge3
from test_delaunay3 import right_edge as right_edge3
import time, copy, os

tree2 = domain_decomp.tree("kdtree", pts2, left_edge2, right_edge2,
                           periodic=False, leafsize=pts2.shape[0]/2 + 2)
tree3 = domain_decomp.tree("kdtree", pts3, left_edge3, right_edge3,
                           periodic=False, leafsize=pts3.shape[0]/2 + 2)
tree2_periodic = domain_decomp.tree("kdtree", pts2, left_edge2, right_edge2,
                                    periodic=True, leafsize=pts2.shape[0]/2 + 2)
tree3_periodic = domain_decomp.tree("kdtree", pts3, left_edge3, right_edge3,
                                    periodic=True, leafsize=pts3.shape[0]/2 + 2)
assert(tree2.num_leaves == 2)
assert(tree3.num_leaves == 2)

def test_ParallelLeaf_2D():
    tree = tree2 ; pts = pts2
    out = []
    pleaves = []
    for i,leaf in enumerate(tree.leaves):
        assert(leaf.id == i)
        pleaf = parallel.ParallelLeaf(leaf)
        pleaf.tessellate(pts2)
        pleaves.append(pleaf)
        out.append(pleaf.outgoing_points())
    pleaves[0].incoming_points(1, out[1][0], pts[out[1][0],:])
    pleaves[0].incoming_points(0, out[0][0], pts[out[0][0],:])

def test_ParallelLeaf_3D():
    tree = tree3 ; pts = pts3
    out = []
    pleaves = []
    for i,leaf in enumerate(tree.leaves):
        assert(leaf.id == i)
        pleaf = parallel.ParallelLeaf(leaf)
        pleaf.tessellate(pts)
        pleaves.append(pleaf)
        out.append(pleaf.outgoing_points())
    pleaves[0].incoming_points(1, out[1][0], pts[out[1][0],:])
    pleaves[0].incoming_points(0, out[0][0], pts[out[0][0],:])

def test_ParallelLeaf_periodic_2D():
    tree = tree2_periodic ; pts = pts2
    out = []
    pleaves = []
    for i,leaf in enumerate(tree.leaves):
        assert(leaf.id == i)
        pleaf = parallel.ParallelLeaf(leaf)
        pleaf.tessellate(pts)
        pleaves.append(pleaf)
        out.append(pleaf.outgoing_points())
    pleaves[0].incoming_points(1, out[1][0], pts[out[1][0],:])
    pleaves[0].incoming_points(0, out[0][0], pts[out[0][0],:])

def test_ParallelLeaf_periodic_3D():
    tree = tree3_periodic ; pts = pts3
    out = []
    pleaves = []
    for i,leaf in enumerate(tree.leaves):
        assert(leaf.id == i)
        pleaf = parallel.ParallelLeaf(leaf)
        pleaf.tessellate(pts)
        pleaves.append(pleaf)
        out.append(pleaf.outgoing_points())
    pleaves[0].incoming_points(1, out[1][0], pts[out[1][0],:])
    pleaves[0].incoming_points(0, out[0][0], pts[out[0][0],:])

def test_ParallelDelaunay_2D():
    tree = tree2 ; pts = pts2
    T_para = parallel.ParallelDelaunay(pts, tree, 2)
    # fname_test = "test_parallel_plot2D.png"
    # axs = T_para.plot(plotfile=fname_test, title='Test')
    # os.remove(fname_test)
    T_seri = delaunay.Delaunay(pts)
    assert(T_para.is_equivalent(T_seri))

def test_ParallelDelaunay_3D():
    tree = tree3 ; pts = pts3
    T_seri = delaunay.Delaunay(pts)
    T_para = parallel.ParallelDelaunay(pts, tree, 2)
    # for name,T in zip(["Serial:","Parallel:"],[T_seri,T_para]):
    #     print(name)
    #     print('    verts',T.num_finite_verts, T.num_infinite_verts, T.num_verts)
    #     print('    cells',T.num_finite_cells, T.num_infinite_cells, T.num_cells)
    #     print('    edges',T.num_finite_edges, T.num_infinite_edges, T.num_edges)
    #     print('    facets',T.num_finite_facets, T.num_infinite_facets, T.num_facets)
    # print(T_para.is_equivalent(T_seri))
    assert(T_para.is_equivalent(T_seri))

# def test_ParallelDelaunay_periodic_2D():
#     leaves2_periodic = copy.deepcopy(leaves20_periodic)
#     T2_para = parallel.ParallelDelaunay(pts2, leaves2_periodic, 2)
#     T2_seri = delaunay.Delaunay(pts2)
#     assert(T2_para.is_equivalent(T2_seri))

# def test_ParallelDelaunay_periodic_3D():
#     leaves3_periodic = copy.deepcopy(leaves30_periodic)
#     T3_para = parallel.ParallelDelaunay(pts3, leaves3_periodic, 2)
#     T3_seri = delaunay.Delaunay(pts3)
#     assert(T3_para.is_equivalent(T3_seri))

def test_DelaunayProcess2():
    pts = pts2 ; tree = tree2
    leaves = tree.leaves
    # leaves = copy.deepcopy(leaves20)
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

