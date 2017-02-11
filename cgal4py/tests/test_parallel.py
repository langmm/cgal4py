r"""Tests for parallel implementation of triangulations."""
import nose.tools as nt
import numpy as np
import os
import time
from cgal4py import _use_multiprocessing
from cgal4py import parallel, delaunay
from cgal4py.domain_decomp import GenericTree
from cgal4py.tests.test_cgal4py import make_points, make_test, MyTestCase
if _use_multiprocessing:
    import multiprocessing as mp
from mpi4py import MPI
import ctypes
np.random.seed(10)


@nt.nottest
def lines_load_test(npts, ndim, periodic=False):
    lines = [
        "from cgal4py.tests.test_cgal4py import make_points",
        "pts, le, re = make_points({}, {})".format(npts, ndim),
        "load_dict = dict(pts=pts, left_edge=le, right_edge=re,",
        "                 periodic={})".format(periodic)]
    return lines


class TestGetMPIType(MyTestCase):

    def setup_param(self):
        self._func = parallel._get_mpi_type
        self.param_equal = [(MPI.INT, ['i'], {}),
                            (MPI.LONG, ['l'], {}),
                            (MPI.FLOAT, ['f'], {}),
                            (MPI.DOUBLE, ['d'], {})]
        self.param_raises = [(ValueError, ['m'], {})]


class TestWriteMPIScript(MyTestCase):

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
        time.sleep(1)
        self.func(self._fname, self._read_lines, 'volumes', overwrite=False)
        t1 = os.path.getmtime(self._fname)
        nt.eq_(t0, t1)
        time.sleep(1)
        self.func(self._fname, self._read_lines, 'volumes', overwrite=True)
        t2 = os.path.getmtime(self._fname)
        nt.assert_not_equal(t1, t2)
        os.remove(self._fname)


class TestParallelLeaf(MyTestCase):

    def setup_param(self):
        self._func = parallel.ParallelLeaf
        self.param_runs = [
            ((0, 2), {}),
            ((0, 3), {}),
            # ((0, 4), {}),
            ((0, 2), {'periodic':True}),
            ((0, 3), {'periodic':True}),
            # ((0, 4), {'periodic':True}),
            ]

    def check_runs(self, args, kwargs):
        pts, tree = make_test(*args, **kwargs)
        left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
        right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        for leaf in tree.leaves:
            pleaf = self._func(leaf, left_edges, right_edges)

    def check_tessellate(self, args, kwargs):
        pts, tree = make_test(*args, **kwargs)
        left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
        right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        leaf = tree.leaves[0]
        pleaf = self._func(leaf, left_edges, right_edges)
        pleaf.pts = pts[tree.idx[pleaf.idx], :]
        pleaf.tessellate()
        pleaf.tessellate(pts)
        pleaf.tessellate(pts, tree.idx)

    def check_exchange(self, args, kwargs):
        pts, tree = make_test(*args, **kwargs)
        left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
        right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        leaf0 = tree.leaves[0]
        leaf1 = tree.leaves[1]
        pleaf0 = self._func(leaf0, left_edges, right_edges)
        pleaf1 = self._func(leaf1, left_edges, right_edges)
        pleaf0.tessellate(pts, tree.idx)
        pleaf1.tessellate(pts, tree.idx)
        out0 = pleaf0.outgoing_points()
        out1 = pleaf1.outgoing_points()
        pleaf1.incoming_points(0, out0[0][1], out0[1], out0[2], out0[3],
                               pts[out0[0][1], :])
        pleaf1.incoming_points(1, out1[0][1], out1[1], out1[2], out1[3],
                               pts[out1[0][1], :])
        pleaf0.incoming_points(0, out0[0][0], out0[1], out0[2], out0[3],
                               pts[out0[0][0], :])
        pleaf0.incoming_points(1, out1[0][0], out1[1], out1[2], out1[3],
                               pts[out1[0][0], :])
        if kwargs.get('periodic', True):
            idx = pleaf1.idx
            pos = pts[idx, :]
            pleaf1.periodic_left[0] = True
            pleaf1.periodic_right[0] = True
            pleaf1.left_neighbors.append(0)
            pos[0,0] = tree.left_edge[0]
            pos[1,0] = tree.right_edge[0]
            pleaf1.incoming_points(1, idx, pleaf1.neighbors, 
                                   pleaf1.left_edges, pleaf1.right_edges, pos)
            pleaf1.incoming_points(0, idx, pleaf0.neighbors, 
                                   pleaf0.left_edges, pleaf0.right_edges, pos)

    def test_tessellate_generator(self):
        for args, kwargs in self.param_runs:
            yield self.check_tessellate, args, kwargs

    def test_exchange_generator(self):
        for args, kwargs in self.param_runs:
            yield self.check_exchange, args, kwargs

    def test_serialize(self):
        pts, tree = make_test(0, 2)
        left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
        right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
        leaf = tree.leaves[0]
        pleaf = self._func(leaf, left_edges, right_edges)
        pleaf.tessellate(pts, tree.idx)
        pleaf.serialize(store=True)


class TestDelaunayProcessMPI(MyTestCase):

    def setup_param(self):
        self._func = parallel.DelaunayProcessMPI
        taskname1 = 'triangulate'
        taskname2 = 'volumes'
        ndim = 2
        periodic = False
        pts, tree = make_test(0, ndim, periodic=periodic)
        le = tree.left_edge
        re = tree.right_edge
        self._pts = pts
        self._tree = tree
        # self._dummy3 = self._func(taskname2, pts, tree)
        # self._dummy4 = self._func(taskname2, pts, tree, use_buffer=True)
        # self._leaves = self._dummy1._leaves
        # TODO: use_buffer curregntly segfaults when run with coverage
        self.param_runs = [
            # Using C++ communications
            # ((taskname1, pts), {}),
            # ((taskname1, pts, left_edge=le, right_edge=re), {}),
            # ((taskname1, pts), {'use_double':True}),
            # ((taskname1, pts), {'limit_mem':True}),
            # ((taskname2, pts), {}),
            # ((taskname2, pts), {'limit_mem':True}),
            # Using Python communications
            ((taskname1, pts), {'use_python':True}),
            ((taskname1, pts, tree), {'use_python':True}),
            ((taskname1, pts, GenericTree.from_tree(tree)),
                 {'use_python':True}),
            ((taskname1, pts, tree), {'use_python':True,
                                      'use_double':True}),
            # ((taskname1, pts, tree), {'use_python':True},
            #                           'use_buffer':True}),
            ((taskname1, pts, tree), {'use_python':True,
                                      'limit_mem':True}),
            ((taskname2, pts, tree), {'use_python':True}),
            # ((taskname2, pts, tree), {'use_python':True},
            #                           'use_buffer':True}),
            ((taskname2, pts, tree), {'use_python':True,
                                      'limit_mem':True}),
            ]
        self.param_raises = [
            (ValueError, ('null', pts, tree), {})
            ]

    def check_runs(self, args, kwargs):
        x = self.func(*args, **kwargs)
        x.run()
        fname = x.output_filename()
        if os.path.isfile(fname):
            os.remove(fname)

    def test_gather_leaf_arrays(self):
        taskname1 = 'triangulate'
        pts = self._pts
        tree = self._tree
        dummy1 = self.func(taskname1, pts, tree, use_python=True)
        dummy2 = self.func(taskname1, pts, tree, use_python=True,
                           use_buffer=True)
        leaves = tree.leaves
        arr = {leaf.id: np.arange(5*(leaf.id+1)) for leaf in leaves}
        dummy1.gather_leaf_arrays(arr)
        dummy2.gather_leaf_arrays(arr)


if _use_multiprocessing:
    class TestDelaunayProcessMulti(MyTestCase):

        def setup_param(self):
            self._func = parallel.DelaunayProcessMulti
            self.param_runs = [
                (('triangulate',), {}),
                (('triangulate',), {'limit_mem':True}),
                (('volumes',), {}),
                (('triangulate',), {'limit_mem':True}),
                ]

        def check_runs(self, args, kwargs):
            (taskname,) = args
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
            nproc = 2  # len(leaves)
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
            processes = []
            for i in xrange(nproc):
                processes.append(self._func(
                    taskname, i, task2leaves[i],
                    ptsArray, idxArray, left_edges, right_edges,
                    queues, lock, count, in_pipes[i], **kwargs))
            # Perform setup on higher processes
            for i in xrange(1, nproc):
                P = processes[i]
                P.tessellate_leaves()
                P.outgoing_points()
            for i in xrange(1, nproc):
                count[0].value += 1
            # Perform entire run on lowest process
            P = processes[0]
            P.run(test_in_serial=True)
            # Do tear down on higher processes
            for i in xrange(1, nproc):
                P = processes[i]
                P.incoming_points()
                P.enqueue_result()
                for l in range(len(task2leaves[i])):
                    x = P.receive_result(out_pipes[i])
            # Clean up files
            for i in xrange(nproc):
                P = processes[i]
                for leaf in P._leaves:
                    if kwargs.get('limit_mem', False):
                        leaf.remove_tess()
                    ffinal = leaf.tess_output_filename
                    if os.path.isfile(ffinal):
                        os.remove(ffinal)


class TestParallelDelaunay(MyTestCase):

    def setup_param(self):
        self._func = parallel.ParallelDelaunay
        ndim_list = [2, 3]  # , 4]
        param_test = []
        self._fprof = 'test_ParallelDelaunay.cProfile'
        for ndim in ndim_list:
            param_test += [
                ((0, ndim, 2), {}),
                ((100, ndim, 2), {'nleaves': 2}),
                ((100, ndim, 4), {'nleaves': 4}),
                ((100, ndim, 4), {'nleaves': 8}),
                # ((1000, ndim, 2), {'nleaves': 2}),
                # ((4*4*2, ndim, 4), {'leafsize': 8}),
                # ((1e5, ndim, 8), {'nleaves': 8}),
                # ((1e7, ndim, 10), {'nleaves': 10}),
                ]
        self.param_returns = []
        for args, kwargs in param_test:
            pts, tree = make_test(args[0], args[1], **kwargs)
            ans = delaunay.Delaunay(pts)
            read_lines = lines_load_test(args[0], args[1])
            for limit_mem in [False, True]:
                if _use_multiprocessing:
                    self.param_returns += [
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': False, 'limit_mem': limit_mem})
                        ]
                for profile in [False, self._fprof]:
                    self.param_returns += [
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_python':True,
                              'use_buffer': False}),
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_python':True,
                              'use_buffer': True}),
                        # (ans, (pts, tree, args[2]),
                        #      {'use_mpi': True, 'limit_mem': limit_mem,
                        #       'profile': profile, 'use_python':False})
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
                        T.num_facets, T.num_finite_facets,
                        T.num_infinite_facets))
            raise
        if os.path.isfile(self._fprof):
            os.remove(self._fprof)


class TestParallelVoronoiVolumes(MyTestCase):

    def setup_param(self):
        self._func = parallel.ParallelVoronoiVolumes
        ndim_list = [2, 3]  # , 4]
        param_test = []
        self._fprof = 'test_ParallelVoronoiVolumes.cProfile'
        for ndim in ndim_list:
            param_test += [
                ((0, ndim, 2), {}),
                ((100, ndim, 2), {'nleaves': 2}),
                ((100, ndim, 4), {'nleaves': 4}),
                ((100, ndim, 4), {'nleaves': 8}),
                # ((1000, ndim, 2), {'nleaves': 2}),
                # ((4*4*2, ndim, 4), {'leafsize': 8}),
                # ((1e5, ndim, 8), {'nleaves': 8}),
                # ((1e7, ndim, 10), {'nleaves': 10}),
                ]
        self.param_returns = []
        for args, kwargs in param_test:
            pts, tree = make_test(args[0], args[1], **kwargs)
            ans = delaunay.VoronoiVolumes(pts)
            read_lines = lines_load_test(args[0], args[1])
            for limit_mem in [False, True]:
                if _use_multiprocessing:
                    self.param_returns += [
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': False, 'limit_mem': limit_mem})
                        ]
                for profile in [False, self._fprof]:
                    self.param_returns += [
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_python':True,
                              'use_buffer': False}),
                        (ans, (pts, tree, args[2]),
                             {'use_mpi': True, 'limit_mem': limit_mem,
                              'profile': profile, 'use_python':True,
                              'use_buffer': True}),
                        # (ans, (pts, tree, args[2]),
                        #      {'use_mpi': True, 'limit_mem': limit_mem,
                        #       'profile': profile, 'use_python':False})
                        ]

    def check_returns(self, result, args, kwargs):
        print(result)
        print(self.func(*args, **kwargs))
        assert(np.allclose(result, self.func(*args, **kwargs)))
        if os.path.isfile(self._fprof):
            os.remove(self._fprof)
