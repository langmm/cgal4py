r"""Tests for parallel implementation of triangulations."""
import numpy as np
from nose.tools import nottest
from cgal4py import parallel, delaunay
from test_cgal4py import make_points, make_test
from test_delaunay2 import pts as pts2
np.random.seed(10)


@nottest
def lines_load_test(npts, ndim, periodic=False):
    lines = [
        "from cgal4py.tests.test_cgal4py import make_points",
        "pts, le, re = make_points({}, {})".format(npts, ndim),
        "load_dict = dict(pts=pts, left_edge=le, right_edge=re,",
        "                 periodic={})".format(periodic)]
    return lines


@nottest
def runtest_ParallelVoronoiVolumes(npts, ndim, nproc, use_mpi=False,
                                   use_buffer=False, profile=False, **kwargs):
    pts, tree = make_test(npts, ndim, **kwargs)
    v_seri = delaunay.VoronoiVolumes(pts)
    if use_mpi:
        v_para = parallel.ParallelVoronoiVolumesMPI(
            lines_load_test(npts, ndim), ndim, nproc, use_buffer=use_buffer,
            profile=profile)
    else:
        v_para = parallel.ParallelVoronoiVolumes(pts, tree, nproc)
    assert(np.allclose(v_seri, v_para))


@nottest
def runtest_ParallelDelaunay(npts, ndim, nproc, use_mpi=False,
                             use_buffer=False, profile=False, **kwargs):
    pts, tree = make_test(npts, ndim, **kwargs)
    T_seri = delaunay.Delaunay(pts)
    try:
        if use_mpi:
            T_para = parallel.ParallelDelaunayMPI(lines_load_test(npts, ndim),
                                                  ndim, nproc,
                                                  use_buffer=use_buffer,
                                                  profile=profile)
        else:
            T_para = parallel.ParallelDelaunay(pts, tree, nproc)
    except:
        print(("Test failed with npts={}, ndim={}, nproc={}, use_mpi={}, " +
               "use_buffer={}").format(npts, ndim, nproc, use_mpi, use_buffer))
        raise
    c_seri, n_seri, inf_seri = T_seri.serialize(sort=True)
    c_para, n_para, inf_para = T_para.serialize(sort=True)
    try:
        assert(np.all(c_seri == c_para))
        assert(np.all(n_seri == n_para))
        assert(T_para.is_equivalent(T_seri))
    except:
        for name, T in zip(['Parallel','Serial'],[T_para, T_seri]):
            print(name)
            print('\t verts: {}, {}, {}'.format(
                T.num_verts, T.num_finite_verts, T.num_infinite_verts))
            print('\t cells: {}, {}, {}'.format(
                T.num_cells, T.num_finite_cells, T.num_infinite_cells))
            print('\t edges: {}, {}, {}'.format(
                T.num_edges, T.num_finite_edges, T.num_infinite_edges))
            if ndim == 3:
                print('\t facets: {}, {}, {}'.format(
                    T.num_facets, T.num_finite_facets, T.num_infinite_facets))
        raise


@nottest
def runtest(func_name, *args, **kwargs):
    if func_name in ['delaunay', 'Delaunay', 'triangulate']:
        func = runtest_ParallelDelaunay
    elif func_name in ['volumes','VoronoiVolumes']:
        func = runtest_ParallelVoronoiVolumes
    else:
        raise ValueError("Unrecognized test function: {}".format(func_name))
    return func(*args, **kwargs)


@nottest
def runtest_ParallelSeries(func_name, ndim, use_mpi=False, periodic=False,
                           profile=False):
    tests = [
        {'npts':0, 'nproc':2, 'kwargs': {}},
        {'npts':1000, 'nproc':2, 'kwargs': {'nleaves': 2}},
        {'npts':4*4*2, 'nproc':4, 'kwargs': {'leafsize': 8}},
        {'npts':1e5, 'nproc':8, 'kwargs': {'nleaves': 8}},
        # {'npts':1e7, 'nproc':10, 'kwargs': {'nleaves': 10}},
    ]
    if ndim > 2:
        tests = tests[1:-1]
    for t in tests:
        runtest(t['npts'], ndim, t['nproc'], use_mpi=use_mpi,
                periodic=periodic, profile=profile, **t['kwargs'])
    if use_mpi:
        for t in tests:
            runtest(t['npts'], ndim, t['nproc'], use_mpi=use_mpi,
                    use_buffer=True, periodic=periodic, profile=profile,
                    **t['kwargs'])


# def test_ParallelLeaf_2D():
#     pts, tree = make_test(0, 2)
#     left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
#     right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
#     out = []
#     pleaves = []
#     for i, leaf in enumerate(tree.leaves):
#         assert(leaf.id == i)
#         pleaf = parallel.ParallelLeaf(leaf, left_edges, right_edges)
#         pleaf.tessellate(pts2)
#         pleaves.append(pleaf)
#         out.append(pleaf.outgoing_points())
#     pleaves[0].incoming_points(1, out[1][0][0], out[1][1], out[1][2],
#                                out[1][3], pts[out[1][0][0], :])
#     pleaves[0].incoming_points(0, out[0][0][0], out[0][1], out[0][2],
#                                out[0][3], pts[out[0][0][0], :])


# def test_ParallelLeaf_3D():
#     pts, tree = make_test(0, 3)
#     left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
#     right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
#     out = []
#     pleaves = []
#     for i, leaf in enumerate(tree.leaves):
#         assert(leaf.id == i)
#         pleaf = parallel.ParallelLeaf(leaf, left_edges, right_edges)
#         pleaf.tessellate(pts)
#         pleaves.append(pleaf)
#         out.append(pleaf.outgoing_points())
#     pleaves[0].incoming_points(1, out[1][0][0], out[1][1], out[1][2],
#                                out[1][3], pts[out[1][0][0], :])
#     pleaves[0].incoming_points(0, out[0][0][0], out[0][1], out[0][2],
#                                out[0][3], pts[out[0][0][0], :])


# def test_ParallelLeaf_periodic_2D():
#     pts, tree = make_test(0, 2, periodic=True)
#     left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
#     right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
#     out = []
#     pleaves = []
#     for i, leaf in enumerate(tree.leaves):
#         assert(leaf.id == i)
#         pleaf = parallel.ParallelLeaf(leaf, left_edges, right_edges)
#         pleaf.tessellate(pts)
#         pleaves.append(pleaf)
#         out.append(pleaf.outgoing_points())
#     pleaves[0].incoming_points(1, out[1][0][0], out[1][1], out[1][2],
#                                out[1][3], pts[out[1][0][0], :])
#     pleaves[0].incoming_points(0, out[0][0][0], out[0][1], out[0][2],
#                                out[0][3], pts[out[0][0][0], :])


# def test_ParallelLeaf_periodic_3D():
#     pts, tree = make_test(0, 3, periodic=True)
#     left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
#     right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
#     out = []
#     pleaves = []
#     for i, leaf in enumerate(tree.leaves):
#         assert(leaf.id == i)
#         pleaf = parallel.ParallelLeaf(leaf, left_edges, right_edges)
#         pleaf.tessellate(pts)
#         pleaves.append(pleaf)
#         out.append(pleaf.outgoing_points())
#     pleaves[0].incoming_points(1, out[1][0][0], out[1][1], out[1][2],
#                                out[1][3], pts[out[1][0][0], :])
#     pleaves[0].incoming_points(0, out[0][0][0], out[0][1], out[0][2],
#                                out[0][3], pts[out[0][0][0], :])


def test_ParallelVoronoiVolumes():
    # 2D
    ndim = 2
    runtest_ParallelSeries('volumes', ndim)
    runtest_ParallelSeries('volumes', ndim, periodic=True)
    runtest_ParallelSeries('volumes', ndim, use_mpi=True)
    runtest_ParallelSeries('volumes', ndim, use_mpi=True, periodic=True)
    # 3D
    ndim = 3
    runtest_ParallelSeries('volumes', ndim)
    runtest_ParallelSeries('volumes', ndim, periodic=True)
    runtest_ParallelSeries('volumes', ndim, use_mpi=True)
    runtest_ParallelSeries('volumes', ndim, use_mpi=True, periodic=True)


def test_ParallelDelaunay():
    # 2D
    ndim = 2
    runtest_ParallelSeries('delaunay', ndim)
    runtest_ParallelSeries('delaunay', ndim, periodic=True)
    runtest_ParallelSeries('delaunay', ndim, use_mpi=True)
    runtest_ParallelSeries('delaunay', ndim, use_mpi=True, periodic=True)
    # 3D
    ndim = 3
    runtest_ParallelSeries('delaunay', ndim)
    runtest_ParallelSeries('delaunay', ndim, periodic=True)
    runtest_ParallelSeries('delaunay', ndim, use_mpi=True)
    runtest_ParallelSeries('delaunay', ndim, use_mpi=True, periodic=True)


# def test_DelaunayProcess2():
#     pts, tree = make_test(0, 2)
#     idxArray = mp.RawArray(ctypes.c_ulonglong, tree.idx.size)
#     ptsArray = mp.RawArray('d',pts.size)
#     memoryview(idxArray)[:] = tree.idx
#     memoryview(ptsArray)[:] = pts
#     left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
#     right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
#     leaves = tree.leaves
#     nproc = 2 # len(leaves)
#     count = [mp.Value('i',0),mp.Value('i',0),mp.Value('i',0)]
#     lock = mp.Condition()
#     queues = [mp.Queue() for _ in xrange(nproc+1)]
#     in_pipes = [None for _ in xrange(nproc)]
#     out_pipes = [None for _ in xrange(nproc)]
#     for i in range(nproc):
#         out_pipes[i],in_pipes[i] = mp.Pipe(True)
#     # Split leaves
#     task2leaves = [[] for _ in xrange(nproc)]
#     for leaf in leaves:
#         task = leaf.id % nproc
#         task2leaves[task].append(leaf)
#     # Create processes & tessellate
#     processes = []
#     for i in xrange(nproc):
#         P = parallel.DelaunayProcess('triangulate', i, task2leaves[i],
#                                      ptsArray, idxArray,
#                                      left_edges, right_edges,
#                                      queues, lock, count, in_pipes[i])
#         processes.append(P)
#     # Split
#     P1, P2 = processes[0], processes[1]
#     # Do partial run on 1
#     P1.tessellate_leaves()
#     P1.outgoing_points()
#     # Full run on 2
#     P2.run()
#     # Finish on 1
#     i,j,arr,ln,rn = queues[0].get()
#     queues[0].put((i,j,np.array([]),ln,rn))
#     P1.incoming_points()
#     P1.enqueue_triangulation()
#     time.sleep(0.01)
