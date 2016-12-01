r"""Tests for parallel implementation of triangulations."""
import nose.tools as nt
import unittest
import numpy as np
import os
from nose.tools import assert_raises, nottest
from cgal4py import parallel, delaunay
from cgal4py.domain_decomp import GenericTree
from test_cgal4py import make_points, make_test
from test_delaunay2 import pts as pts2
import multiprocessing as mp
from mpi4py import MPI
import ctypes
np.random.seed(10)


@nottest
def lines_load_test(npts, ndim, periodic=False):
    lines = [
        "from cgal4py.tests.test_cgal4py import make_points",
        "pts, le, re = make_points({}, {})".format(npts, ndim),
        "load_dict = dict(pts=pts, left_edge=le, right_edge=re,",
        "                 periodic={})".format(periodic)]
    return lines


# @nottest
# def runtest(func_name, *args, **kwargs):
#     if func_name in ['delaunay', 'Delaunay', 'triangulate']:
#         func = runtest_ParallelDelaunay
#     elif func_name in ['volumes','VoronoiVolumes']:
#         func = runtest_ParallelVoronoiVolumes
#     else:
#         raise ValueError("Unrecognized test function: {}".format(func_name))
#     return func(*args, **kwargs)


# @nottest
# def runtest_ParallelSeries(func_name, ndim, use_mpi=False, periodic=False,
#                            profile=False):
#     tests = [
#         {'npts':0, 'nproc':2, 'kwargs': {}},
#         {'npts':1000, 'nproc':2, 'kwargs': {'nleaves': 2}},
#         {'npts':4*4*2, 'nproc':4, 'kwargs': {'leafsize': 8}},
#         # {'npts':1e5, 'nproc':8, 'kwargs': {'nleaves': 8}},
#         # {'npts':1e7, 'nproc':10, 'kwargs': {'nleaves': 10}},
#     ]
#     if ndim > 2:
#         tests = tests[1:-1]
#     for t in tests:
#         runtest(func_name, t['npts'], ndim, t['nproc'], use_mpi=use_mpi,
#                 periodic=periodic, profile=profile, **t['kwargs'])
#     if use_mpi:
#         for t in tests:
#             runtest(func_name, t['npts'], ndim, t['nproc'], use_mpi=use_mpi,
#                     use_buffer=True, periodic=periodic, profile=profile,
#                     **t['kwargs'])


# @nottest
# def runtest_ParallelVoronoiVolumes(npts, ndim, nproc, use_mpi=False,
#                                    use_buffer=False, profile=False, **kwargs):
#     pts, tree = make_test(npts, ndim, **kwargs)
#     v_seri = delaunay.VoronoiVolumes(pts)
#     if use_mpi:
#         v_para = parallel.ParallelVoronoiVolumesMPI(
#             lines_load_test(npts, ndim), ndim, nproc, use_buffer=use_buffer,
#             profile=profile)
#     else:
#         v_para = parallel.ParallelVoronoiVolumes(pts, tree, nproc)
#     assert(np.allclose(v_seri, v_para))


# @nottest
# def runtest_ParallelDelaunay(npts, ndim, nproc, use_mpi=False,
#                              use_buffer=False, profile=False, **kwargs):
#     pts, tree = make_test(npts, ndim, **kwargs)
#     T_seri = delaunay.Delaunay(pts)
#     try:
#         if use_mpi:
#             T_para = parallel.ParallelDelaunayMPI(lines_load_test(npts, ndim),
#                                                   ndim, nproc,
#                                                   use_buffer=use_buffer,
#                                                   profile=profile)
#         else:
#             T_para = parallel.ParallelDelaunay(pts, tree, nproc)
#     except:
#         print(("Test failed with npts={}, ndim={}, nproc={}, use_mpi={}, " +
#                "use_buffer={}").format(npts, ndim, nproc, use_mpi, use_buffer))
#         raise
#     c_seri, n_seri, inf_seri = T_seri.serialize(sort=True)
#     c_para, n_para, inf_para = T_para.serialize(sort=True)
#     try:
#         assert(np.all(c_seri == c_para))
#         assert(np.all(n_seri == n_para))
#         assert(T_para.is_equivalent(T_seri))
#     except:
#         for name, T in zip(['Parallel','Serial'],[T_para, T_seri]):
#             print(name)
#             print('\t verts: {}, {}, {}'.format(
#                 T.num_verts, T.num_finite_verts, T.num_infinite_verts))
#             print('\t cells: {}, {}, {}'.format(
#                 T.num_cells, T.num_finite_cells, T.num_infinite_cells))
#             print('\t edges: {}, {}, {}'.format(
#                 T.num_edges, T.num_finite_edges, T.num_infinite_edges))
#             if ndim == 3:
#                 print('\t facets: {}, {}, {}'.format(
#                     T.num_facets, T.num_finite_facets, T.num_infinite_facets))
#         raise


class MyTestFunction(object):

    def __init__(self):
        self._func = None
        self.param_runs = []
        self.param_returns = []
        self.param_raises = []
        self.setup_param()

    def setup_param(self):
        pass

    @property
    def func(self):
        if self._func is None:
            raise AttributeError("_func must be set.")
        else:
            return self._func

    def check_runs(self, args, kwargs):
        self.func(*args, **kwargs)

    def check_returns(self, result, args, kwargs):
        nt.eq_(result, self.func(*args, **kwargs))

    def check_raises(self, excpt, args, kwargs):
        nt.assert_raises(excpt, self.func, *args, **kwargs)

    def test_runs_generator(self):
        for args, kwargs in self.param_runs:
            yield self.check_runs, args, kwargs

    def test_returns_generator(self):
        for res, args, kwargs in self.param_returns:
            yield self.check_returns, res, args, kwargs

    def test_raises_generator(self):
        for err, args, kwargs in self.param_raises:
            yield self.check_raises, err, args, kwargs


class TestGetMPIType(MyTestFunction):

    def setup_param(self):
        self._func = parallel._get_mpi_type
        self.param_equal = [(MPI.INT, ['i'], {}),
                            (MPI.LONG, ['l'], {}),
                            (MPI.FLOAT, ['f'], {}),
                            (MPI.DOUBLE, ['d'], {})]
        self.param_raises = [(ValueError, ['m'], {})]


class TestWriteMPIScript(MyTestFunction):

    def setup_param(self):
        self._func = parallel.write_mpi_script
        fname = 'test_mpi_script.py'
        read_lines = lines_load_test(10, 2)
        self.param_runs = [
            ((fname, read_lines, 'triangulate'), {}),
            ((fname, read_lines, 'triangulate'), dict(use_double=True)),
            ((fname, read_lines, 'triangulate'), dict(use_buffer=True)),
            ((fname, read_lines, 'triangulate'), dict(profile=True))]
        self._fname = fname
        self._read_lines = read_lines

    def check_runs(self, args, kwargs):
        self.func(*args, **kwargs)
        assert(os.path.isfile(args[0]))
        os.remove(args[0])

    def test_overwrite(self):
        self.func(self._fname, self._read_lines, 'volumes')
        t0 = os.path.getmtime(self._fname)
        self.func(self._fname, self._read_lines, 'volumes', overwrite=False)
        t1 = os.path.getmtime(self._fname)
        nt.eq_(t0, t1)
        self.func(self._fname, self._read_lines, 'volumes', overwrite=True)
        t2 = os.path.getmtime(self._fname)
        nt.assert_not_equal(t1, t2)
        os.remove(self._fname)


class TestParallelLeaf(MyTestFunction):

    def setup_param(self):
        self._func = parallel.ParallelLeaf

    def check_leaves(self, ndim, periodic):
        pts, tree = make_test(0, ndim, periodic=periodic)
        left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
        right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        out = []
        pleaves = []
        for i, leaf in enumerate(tree.leaves):
            assert(leaf.id == i)
            pleaf = parallel.ParallelLeaf(leaf, left_edges, right_edges)
            pleaf.tessellate(pts)
            pleaves.append(pleaf)
            out.append(pleaf.outgoing_points())
        pleaves[0].incoming_points(1, out[1][0][0], out[1][1], out[1][2],
                                   out[1][3], pts[out[1][0][0], :])
        pleaves[0].incoming_points(0, out[0][0][0], out[0][1], out[0][2],
                                   out[0][3], pts[out[0][0][0], :])

    def test_leaves_generate(self):
        for periodic in [False, True]:
            for ndim in [2, 3]: # ,4]:
                yield self.check_leaves, ndim, periodic


class TestParallelVoronoiVolumes(MyTestFunction):

    def setup_param(self):
        self._func = parallel.ParallelVoronoiVolumes
        ndim_list = [2, 3] # , 4]
        param_test = []
        self._fprof = 'test_ParallelVoronoiVolumes.cProfile'
        for ndim in ndim_list:
            param_test += [
                ((0, ndim, 2), {}),
                ((1000, ndim, 2), {'nleaves': 2}),
                ((4*4*2, ndim, 4), {'leafsize': 8}),
                # ((1e5, ndim, 8), {'nleaves': 8}),
                # ((1e7, ndim, 10), {'nleaves': 10}),
                ]
        self.param_returns = []
        for args, kwargs in param_test:
            pts, tree = make_test(args[0], args[1], **kwargs)
            ans = delaunay.VoronoiVolumes(pts)
            read_lines = lines_load_test(args[0], args[1])
            for limit_mem in [False, True]:
                self.param_returns += [
                    (ans, (pts, tree, args[2]),
                         {'use_mpi': False, 'limit_mem': limit_mem})
                    ]
                for profile in [False, self._fprof]:
                    self.param_returns += [
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_buffer': False}),
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_buffer': True})
                        ]

    def check_returns(self, result, args, kwargs):
        assert(np.allclose(result, self.func(*args, **kwargs)))
        if os.path.isfile(self._fprof):
            os.remove(self._fprof)


class TestParallelDelaunay(MyTestFunction):

    def setup_param(self):
        self._func = parallel.ParallelDelaunay
        ndim_list = [2, 3] # , 4]
        param_test = []
        self._fprof = 'test_ParallelDelaunay.cProfile'
        for ndim in ndim_list:
            param_test += [
                ((0, ndim, 2), {}),
                ((1000, ndim, 2), {'nleaves': 2}),
                ((4*4*2, ndim, 4), {'leafsize': 8}),
                # ((1e5, ndim, 8), {'nleaves': 8}),
                # ((1e7, ndim, 10), {'nleaves': 10}),
                ]
        self.param_returns = []
        for args, kwargs in param_test:
            pts, tree = make_test(args[0], args[1], **kwargs)
            ans = delaunay.Delaunay(pts)
            read_lines = lines_load_test(args[0], args[1])
            for limit_mem in [False, True]:
                self.param_returns += [
                    (ans, (pts, tree, args[2]),
                         {'use_mpi': False, 'limit_mem': limit_mem})
                    ]
                for profile in [False, self._fprof]:
                    self.param_returns += [
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_buffer': False}),
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_buffer': True})
                        ]

    def check_returns(self, result, args, kwargs):
        T_seri = result
        T_para = self.func(*args, **kwargs)
        ndim = args[0].shape[1]
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
        if os.path.isfile(self._fprof):
            os.remove(self._fprof)


class TestDelaunayProcessMPI(MyTestFunction):

    def setup_param(self):
        self._func = parallel.DelaunayProcessMPI
        taskname1 = 'triangulate'
        taskname2 = 'volumes'
        ndim = 2
        periodic = False
        pts, tree = make_test(0, ndim, periodic=periodic)
        self._dummy1 = self._func(taskname1, pts, tree)
        self._dummy2 = self._func(taskname1, pts, tree, use_buffer=True)
        self._dummy3 = self._func(taskname2, pts, tree)
        self._dummy4 = self._func(taskname2, pts, tree, use_buffer=True)
        self._leaves = self._dummy1._leaves
        self.param_runs = [
            ((taskname1, pts, tree), {}),
            ((taskname1, pts, GenericTree.from_tree(tree)), {}),
            ((taskname1, pts, tree), {'use_double':True}),
            ((taskname1, pts, tree), {'use_buffer':True}),
            ((taskname2, pts, tree), {}),
            ((taskname2, pts, tree), {'use_buffer':True})
            ]
        self.param_raises = [
            (ValueError, ('null', pts, tree), {})
            ]

    def check_runs(self, args, kwargs):
        self.func(*args, **kwargs).run()

    def test_get_leaf(self):
        self._dummy1.get_leaf(0)

    def test_tessellate_leaves(self):
        self._dummy1.tessellate_leaves()

    def test_gather_leaf_arrays(self):
        arr = {leaf.id: np.arange(5*(leaf.id+1)) for leaf in self._leaves}
        self._dummy1.gather_leaf_arrays(arr)
        self._dummy2.gather_leaf_arrays(arr)

    def test_alltoall_leaf_arrays(self):
        arr = {(leaf.id, 0): np.arange(5*(leaf.id+1)) for leaf in self._leaves}
        self._dummy1.alltoall_leaf_arrays(arr)
        out = self._dummy2.alltoall_leaf_arrays(arr, return_counts=True)
        self._dummy2.alltoall_leaf_arrays(arr, leaf_counts=out[1])
        self._dummy2.alltoall_leaf_arrays({}, dtype=arr[(0,0)].dtype)
        nt.assert_raises(Exception, self._dummy2.alltoall_leaf_arrays, {})

    def test_outgoing_points(self):
        self._dummy1.tessellate_leaves()
        self._dummy2.tessellate_leaves()
        self._dummy1.outgoing_points()
        self._dummy2.outgoing_points()

    def test_incoming_points(self):
        self._dummy1.tessellate_leaves()
        self._dummy2.tessellate_leaves()
        self._dummy1.outgoing_points()
        self._dummy2.outgoing_points()
        self._dummy1.incoming_points()
        self._dummy2.incoming_points()

    def test_enqueue_triangulation(self):
        self._dummy1.tessellate_leaves()
        self._dummy2.tessellate_leaves()
        self._dummy1.enqueue_triangulation()
        self._dummy2.enqueue_triangulation()
        
    def test_enqueue_volumes(self):
        self._dummy1.tessellate_leaves()
        self._dummy1.enqueue_volumes()
        self._dummy2.tessellate_leaves()
        self._dummy2.enqueue_volumes()


class TestDelaunayProcessMulti(MyTestFunction):

    def setup_param(self):
        self._func = parallel.DelaunayProcessMulti
        taskname = 'triangulate'
        ndim = 2
        periodic = False
        pts, tree = make_test(0, ndim, periodic=periodic)
        idxArray = mp.RawArray(ctypes.c_ulonglong, tree.idx.size)
        ptsArray = mp.RawArray('d', pts.size)
        memoryview(idxArray)[:] = tree.idx
        memoryview(ptsArray)[:] = pts
        left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
        right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        leaves = tree.leaves
        nproc = 2 # len(leaves)
        count = [mp.Value('i',0),mp.Value('i',0),mp.Value('i',0)]
        lock = mp.Condition()
        queues = [mp.Queue() for _ in xrange(nproc+1)]
        in_pipes = [None for _ in xrange(nproc)]
        out_pipes = [None for _ in xrange(nproc)]
        for i in range(nproc):
            out_pipes[i],in_pipes[i] = mp.Pipe(True)
        # Split leaves
        task2leaves = [[] for _ in xrange(nproc)]
        for leaf in leaves:
            task = leaf.id % nproc
            task2leaves[task].append(leaf)
        # Dummy process
        self._dummy_args1 = (taskname, 0, task2leaves[0],
                             ptsArray, idxArray,
                             left_edges, right_edges,
                             queues, lock, count, in_pipes[0])
        self._dummy_args2 = (taskname, 1, task2leaves[1],
                             ptsArray, idxArray,
                             left_edges, right_edges,
                             queues, lock, count, in_pipes[1])
        self._dummy1 = self._func(*self._dummy_args1)
        self._dummy2 = self._func(*self._dummy_args2)
        self.param_runs = []
        self.param_raises = []

    # def check_runs(self, args, kwargs):
    #     self.func(*args, **kwargs).run()

    def test_tessellate_leaves(self):
        self._dummy1.tessellate_leaves()

    # def test_outgoing_points(self):
    #     self._dummy1.outgoing_points()
        

    # def test_incoming_points(self):
    #     self._dummy1.incoming_points()

# # def test_DelaunayProcess2():
# #     pts, tree = make_test(0, 2)
# #     idxArray = mp.RawArray(ctypes.c_ulonglong, tree.idx.size)
# #     ptsArray = mp.RawArray('d',pts.size)
# #     memoryview(idxArray)[:] = tree.idx
# #     memoryview(ptsArray)[:] = pts
# #     left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
# #     right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
# #     leaves = tree.leaves
# #     nproc = 2 # len(leaves)
# #     count = [mp.Value('i',0),mp.Value('i',0),mp.Value('i',0)]
# #     lock = mp.Condition()
# #     queues = [mp.Queue() for _ in xrange(nproc+1)]
# #     in_pipes = [None for _ in xrange(nproc)]
# #     out_pipes = [None for _ in xrange(nproc)]
# #     for i in range(nproc):
# #         out_pipes[i],in_pipes[i] = mp.Pipe(True)
# #     # Split leaves
# #     task2leaves = [[] for _ in xrange(nproc)]
# #     for leaf in leaves:
# #         task = leaf.id % nproc
# #         task2leaves[task].append(leaf)
# #     # Create processes & tessellate
# #     processes = []
# #     for i in xrange(nproc):
# #         P = parallel.DelaunayProcess('triangulate', i, task2leaves[i],
# #                                      ptsArray, idxArray,
# #                                      left_edges, right_edges,
# #                                      queues, lock, count, in_pipes[i])
# #         processes.append(P)
# #     # Split
# #     P1, P2 = processes[0], processes[1]
# #     # Do partial run on 1
# #     P1.tessellate_leaves()
# #     P1.outgoing_points()
# #     # Full run on 2
# #     P2.run()
# #     # Finish on 1
# #     i,j,arr,ln,rn = queues[0].get()
# #     queues[0].put((i,j,np.array([]),ln,rn))
# #     P1.incoming_points()
# #     P1.enqueue_triangulation()
# #     time.sleep(0.01)
