r"""Routines for running triangulations in paralle.

.. todo::
   * parallelism through treading
   * parallelism using MPI

"""
from cgal4py.delaunay import Delaunay, tools
import numpy as np
import os
import time
import copy
import struct
from datetime import datetime
import multiprocessing as mp
import ctypes


def _leaf_tess_filename(leaf_id, unique_str=None):
    if isinstance(unique_str, str):
        fname = '{}_leaf{}.dat'.format(unique_str, leaf_id)
    else:
        fname = 'leaf{}.dat'.format(leaf_id)
    return fname


def ParallelVoronoiVolumes(pts, tree, nproc, use_double=False):
    r"""Return the voronoi cell volumes after constructing triangulation in
    parallel.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional
            coordinates.
        tree (object): Domain decomposition tree for splitting points among the
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        nproc (int): Number of processors that should be used.
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.

    Returns:
        np.ndarray of float64: (n,) array of n voronoi volumes for the provided
            points.

    """
    idxArray = mp.RawArray(ctypes.c_ulonglong, tree.idx.size)
    ptsArray = mp.RawArray('d', pts.size)
    memoryview(idxArray)[:] = tree.idx
    memoryview(ptsArray)[:] = pts
    # pts = pts[tree.idx, :]
    # Split leaves
    task2leaves = [[] for _ in xrange(nproc)]
    for leaf in tree.leaves:
        task = leaf.id % nproc
        task2leaves[task].append(leaf)
    left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
    right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
    # Create & execute processes
    count = [mp.Value('i', 0), mp.Value('i', 0), mp.Value('i')]
    lock = mp.Condition()
    queues = [mp.Queue() for _ in xrange(nproc+1)]
    in_pipes = [None for _ in xrange(nproc)]
    out_pipes = [None for _ in xrange(nproc)]
    for i in range(nproc):
        out_pipes[i], in_pipes[i] = mp.Pipe(True)
    unique_str = datetime.today().strftime("%Y%j%H%M%S")
    processes = [DelaunayProcess(
        'volumes', _, task2leaves[_], ptsArray, idxArray,
        left_edges, right_edges, queues, lock, count, in_pipes[_],
        unique_str=unique_str) for _ in xrange(nproc)]
    for p in processes:
        p.start()
    # Get leaves with tessellation
    vol = np.empty(pts.shape[0], pts.dtype)
    dummy_head = np.empty(1, 'uint64')

    def recv_leaf(p):
        out_pipes[p].recv_bytes_into(dummy_head)
        iid = dummy_head[0]
        assert(tree.leaves[iid].id == iid)
        ivol = np.empty(tree.leaves[iid].npts, 'float64')
        out_pipes[p].recv_bytes_into(ivol)
        vol[tree.idx[tree.leaves[iid].slice]] = ivol

    total_count = 0
    max_total_count = tree.num_leaves
    proc_list = range(nproc)
    t0 = time.time()
    while total_count != max_total_count:
        for i in proc_list:
            while out_pipes[i].poll():
                recv_leaf(i)
                total_count += 1
    # for i in range(nproc):
    #     for _ in range(len(task2leaves[i])):
    #         recv_leaf(i)
    t1 = time.time()
    print("{}s for recieving".format(t1-t0))
    # Cleanup
    for p in processes:
        p.join()
    return vol


def ParallelDelaunay(pts, tree, nproc, use_double=False, in_memory=False):
    r"""Return a triangulation that is constructed in parallel.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional
            coordinates.
        tree (object): Domain decomposition tree for splitting points among the
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        nproc (int): Number of processors that should be used.
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        in_memory (bool, optional): If True, the triangulation results from
            each process are moved to local memory using `multiprocessing`
            pipes. Otherwise, each process writes out tessellation info to
            files which are then incrementally loaded as consolidation occurs.
            Defaults to True.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`: consolidated 2D or 3D
            triangulation object.

    """
    idxArray = mp.RawArray(ctypes.c_ulonglong, tree.idx.size)
    ptsArray = mp.RawArray('d', pts.size)
    memoryview(idxArray)[:] = tree.idx
    memoryview(ptsArray)[:] = pts
    # Split leaves
    task2leaves = [[] for _ in xrange(nproc)]
    for leaf in tree.leaves:
        task = leaf.id % nproc
        task2leaves[task].append(leaf)
    left_edges = np.vstack([leaf.left_edge for leaf in tree.leaves])
    right_edges = np.vstack([leaf.right_edge for leaf in tree.leaves])
    # Create & execute processes
    count = [mp.Value('i', 0), mp.Value('i', 0), mp.Value('i', 0)]
    lock = mp.Condition()
    queues = [mp.Queue() for _ in xrange(nproc+1)]
    in_pipes = [None for _ in xrange(nproc)]
    out_pipes = [None for _ in xrange(nproc)]
    for i in range(nproc):
        out_pipes[i], in_pipes[i] = mp.Pipe(True)
    unique_str = datetime.today().strftime("%Y%j%H%M%S")
    if in_memory:
        processes = [DelaunayProcess(
            'triangulate', _, task2leaves[_], ptsArray, idxArray,
            left_edges, right_edges, queues, lock, count, in_pipes[_],
            unique_str=unique_str) for _ in xrange(nproc)]
    else:
        processes = [DelaunayProcess(
            'output', _, task2leaves[_], ptsArray, idxArray,
            left_edges, right_edges, queues, lock, count, in_pipes[_],
            unique_str=unique_str) for _ in xrange(nproc)]
    for p in processes:
        p.start()
    # Synchronize to ensure rapid receipt of output info from leaves
    lock.acquire()
    lock.wait()
    lock.release()
    # Setup methods for recieving leaf info
    if in_memory:
        serial = [None for _ in xrange(tree.num_leaves)]
        dt2dtype = {0: np.uint32, 1: np.uint64, 2: np.int32, 3: np.int64}
        dummy_head = np.empty(5, 'uint64')

        def recv_leaf(p):
            out_pipes[p].recv_bytes_into(dummy_head)
            iid, ncell, dt, idx_inf, ncell_tot = dummy_head
            dtype = dt2dtype[dt]
            s = [np.empty((ncell, tree.ndim+1), dtype),
                 np.empty((ncell, tree.ndim+1), dtype),
                 dtype(idx_inf),
                 np.empty((ncell, tree.ndim+1), 'uint32'),
                 np.empty(ncell, 'uint64'),
                 ncell_tot]
            for _ in range(2) + range(3, 5):
                out_pipes[p].recv_bytes_into(s[_])
            s = tuple(s)
            assert(tree.leaves[iid].id == iid)
            serial[iid] = s
    else:
        serial = [0]
        dummy_head = np.empty(1, 'uint64')

        def recv_leaf(p):
            out_pipes[p].recv_bytes_into(dummy_head)
            serial[0] += dummy_head[0]
    # Recieve output from processes
    total_count = 0
    max_total_count = tree.num_leaves
    proc_list = range(nproc)
    t0 = time.time()
    while total_count != max_total_count:
        for i in proc_list:
            while out_pipes[i].poll():
                recv_leaf(i)
                total_count += 1
    # for i in range(nproc):
    #     for _ in range(len(task2leaves[i])):
    #         recv_leaf(i)
    t1 = time.time()
    print("Recieving took {} s".format(t1-t0))
    # Consolidate tessellation
    t0 = time.time()
    out = consolidate_tess(tree, serial, pts, use_double=use_double,
                           unique_str=unique_str, in_memory=in_memory)
    t1 = time.time()
    print("Consolidation took {} s".format(t1-t0))
    # Close queues and processes
    for p in processes:
        p.join()
    return out


# @profile
def consolidate_tess(tree, leaf_output, pts, use_double=False,
                     unique_str=None, in_memory=True):
    r"""Creates a single triangulation from the triangulations of leaves.

    Args:
        tree (object): Domain decomposition tree for splitting points among the
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        leaf_output (object): Output from each parallel leaf.
        pts (np.ndarray of float64): (n,m) Array of n mD points.
        use_double (bool, optional): If True, the triangulation is forced to
            use 64bit integers reguardless of if there are too many points for
            32bit. Otherwise 32bit integers are used so long as the number of
            points is <=4294967295. Defaults to False.
        unique_str (str, optional): Unique identifier for files in a run. If
            `in_memory == False` those files will be loaded and used to create
            the consolidated tessellation. Defaults to None. If None, there is
            a risk that multiple runs could be sharing files of the same name.
        in_memory (bool, optional): If True, the triangulation is consolidated
            from partial triangulations on each leaf that already exist in
            memory. Otherwise, partial triangulations are loaded from files for
            each leaf. Defaults to `True`.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`: consolidated 2D or 3D
            triangulation object.

    """
    npts = pts.shape[0]
    ndim = pts.shape[1]
    uint32_max = np.iinfo('uint32').max
    if npts >= uint32_max:
        use_double = True
    if use_double:
        idx_inf = np.uint64(np.iinfo('uint64').max)
    else:
        idx_inf = np.uint32(uint32_max)
    # Loop over leaves adding them
    if in_memory:
        ncells_tot = 0
        for s in leaf_output:
            ncells_tot += np.int64(s[5])
        if use_double:
            cons = tools.ConsolidatedLeaves64(ndim, idx_inf, ncells_tot)
        else:
            cons = tools.ConsolidatedLeaves32(ndim, idx_inf, ncells_tot)
        for i, leaf in enumerate(tree.leaves):
            if isinstance(leaf_output[i][2], np.uint64):
                sleaf = tools.SerializedLeaf64(
                    leaf.id, ndim, leaf_output[i][0].shape[0],
                    leaf_output[i][2], leaf_output[i][0], leaf_output[i][1],
                    leaf_output[i][3], leaf_output[i][4],
                    leaf.start_idx, leaf.stop_idx)
            else:
                sleaf = tools.SerializedLeaf32(
                    leaf.id, ndim, leaf_output[i][0].shape[0],
                    leaf_output[i][2], leaf_output[i][0], leaf_output[i][1],
                    leaf_output[i][3], leaf_output[i][4],
                    leaf.start_idx, leaf.stop_idx)
            cons.add_leaf(sleaf)
    else:
        ncells_tot = leaf_output[0]
        if use_double:
            cons = tools.ConsolidatedLeaves64(ndim, idx_inf, ncells_tot)
        else:
            cons = tools.ConsolidatedLeaves32(ndim, idx_inf, ncells_tot)
        for i, leaf in enumerate(tree.leaves):
            fname = _leaf_tess_filename(leaf.id, unique_str=unique_str)
            cons.add_leaf_fromfile(fname)
            os.remove(fname)
    cons.finalize()
    cells = cons.verts
    neigh = cons.neigh
    # if np.sum(neigh == idx_inf) != 0:
    #     for i in xrange(ncells):
    #         print(i, cells[i, :], neigh[i, :])
    # assert(np.sum(neigh == idx_inf) == 0)
    # Do tessellation
    T = Delaunay(np.zeros([0, ndim]), use_double=use_double)
    T.deserialize_with_info(pts, tree.idx.astype(cells.dtype),
                            cells, neigh, idx_inf)
    return T


class DelaunayProcess(mp.Process):
    r"""`multiprocessing.Process` subclass for coordinating operations on a
    single process during a parallel Delaunay triangulation.

    Args:
        task (str): Key for the task that should be parallelized. Options are:
              'triangulate': Perform triangulation and put serialized info in
                  the output queue.
              'volumes': Perform triangulation and put volumes in output queue.
        proc_idx (int): Index of this process.
        leaves (list of leaf objects): Leaves that should be triangulated on
            this process. The leaves are created by
            :meth:`cgal4py.domain_decomp.tree`.
        pts (np.ndarray of float64): (n,m) array of n m-dimensional
            coordinates. Each leaf has a set of indices identifying coordinates
            within `pts` that belong to that leaf.
        left_edges (np.ndarray float64): Array of mins for all leaves in the
            domain decomposition.
        right_edges (np.ndarray float64): Array of maxes for all leaves in the
            domain decomposition.
        queues (list of `multiprocessing.Queue`): List of queues for every
            process being used in the triangulation plus one for the main
            process.
        lock (multiprocessing.Lock): Lock for processes.
        count (multiprocessing.Value): Shared integer for tracking exchanged
            points.
        pipe (multiprocessing.Pipe): Input end of pipe that is connected to the
            master process.
        unique_str (str, optional): Unique string identifying the domain
            decomposition that is passed to `cgal4py.parallel.ParallelLeaf` for
            file naming. Defaults to None.
        **kwargs: Variable keyword arguments are passed to
            `multiprocessing.Process`.

    Raises:
        ValueError: if `task` is not one of the accepted values listed above.

    """
    def __init__(self, task, proc_idx, leaves, pts, idx,
                 left_edges, right_edges, queues, lock, count, pipe,
                 unique_str=None, **kwargs):
        task_list = ['tessellate', 'exchange', 'enqueue_tess', 'enqueue_vols',
                     'output_tess', 'triangulate', 'volumes', 'output']
        if task not in task_list:
            raise ValueError('{} is not a valid task.'.format(task))
        super(DelaunayProcess, self).__init__(**kwargs)
        self._task = task
        self._leaves = [ParallelLeaf(leaf, left_edges, right_edges, unique_str)
                        for leaf in leaves]
        self._idx = np.frombuffer(idx, dtype='uint64')
        self._ptsFlt = np.frombuffer(pts, dtype='float64')
        ndim = left_edges.shape[1]
        npts = len(self._ptsFlt)/ndim
        self._ndim = ndim
        self._pts = self._ptsFlt.reshape(npts, ndim)
        self._queues = queues
        self._lock = lock
        self._pipe = pipe
        self._count = count
        self._num_proc = len(queues)-1
        self._local_leaves = len(leaves)
        if self._local_leaves == 0:
            self._total_leaves = 0
        else:
            self._total_leaves = leaves[0].num_leaves
        self._proc_idx = proc_idx
        self._done = False

    def tessellate_leaves(self):
        r"""Performs the tessellation for each leaf on this process."""
        for leaf in self._leaves:
            new_pts = np.copy(self._pts[self._idx[leaf.slice], :])
            leaf.tessellate(new_pts)

    def outgoing_points(self):
        r"""Enqueues points at edges of each leaf's boundaries."""
        for leaf in self._leaves:
            hvall, n, le, re = leaf.outgoing_points()
            for i in xrange(self._total_leaves):
                task = i % self._num_proc
                if hvall[i] is None:
                    self._queues[task].put(None)
                else:
                    self._queues[task].put((i, leaf.id, hvall[i], n, le, re))
                time.sleep(0.01)

    def incoming_points(self):
        r"""Takes points from the queue and adds them to the triangulation."""
        queue = self._queues[self._proc_idx]
        count = 0
        nrecv = 0
        while count < (self._local_leaves*self._total_leaves):
            count += 1
            time.sleep(0.01)
            out = queue.get()
            if out is None:
                continue
            i, j, arr, n, le, re = out
            if (arr is None) or (arr.shape[0] == 0):
                continue
            # Find leaf this should go to
            for leaf in self._leaves:
                if leaf.id == i:
                    break
            # Add points to leaves
            new_pts = np.copy(self._pts[self._idx[arr], :])
            leaf.incoming_points(j, arr, n, le, re, new_pts)
            nrecv += arr.shape[0]
        with self._count[1].get_lock():
            self._count[1].value += nrecv

    def enqueue_triangulation(self):
        r"""Enqueue resulting tessellation."""
        for leaf in self._leaves:
            s = leaf.serialize()
            if s[0].dtype == np.uint32:
                dt = 0
            elif s[0].dtype == np.uint64:
                dt = 1
            elif s[0].dtype == np.int32:
                dt = 2
            elif s[0].dtype == np.int64:
                dt = 3
            else:
                raise Exception("No type found for {}".format(s[0].dtype))
            self._pipe.send_bytes(
                struct.pack('QQQQQ', leaf.id, s[0].shape[0], dt, s[2], s[5]))
            for _ in range(2) + range(3, 5):
                self._pipe.send_bytes(s[_])

    def enqueue_volumes(self):
        r"""Enqueue resulting voronoi volumes."""
        for leaf in self._leaves:
            self._pipe.send_bytes(struct.pack('Q', leaf.id))
            self._pipe.send_bytes(leaf.voronoi_volumes())

    def enqueue_number_of_cells(self):
        r"""Enqueue resulting number of cells."""
        ncells = 0
        for leaf in self._leaves:
            leaf.serialize(store=True)
            ncells += leaf.T.num_cells
        self._pipe.send_bytes(struct.pack('Q', ncells))

    def output_tess(self):
        r"""Write serialized tessellation info to file for each leaf."""
        for leaf in self._leaves:
            ncells = leaf.write_tess_to_file()
            self._pipe.send_bytes(struct.pack('Q', ncells))

    def run(self):
        r"""Performs tessellation and communication for each leaf on this
        process."""
        if self._task == 'tessellate':
            self.tessellate_leaves()
        elif self._task == 'exchange':
            self.outgoing_points()
            self.incoming_points()
        elif self._task == 'enqueue_tess':
            self.enqueue_triangulation()
        elif self._task == 'enqueue_vols':
            self.enqueue_volumes()
        elif self._task == 'output_tess':
            self.output_tess()
        elif self._task in ['triangulate', 'volumes', 'output']:
            self.tessellate_leaves()
            # Continue exchanges until there are not any particles that need to
            # be exchanged.
            while True:
                with self._count[2].get_lock():
                    if self._count[2].value == 1:
                        break
                # print('Begin', self._proc_idx, self._count[0].value,
                #       self._count[1].value, self._count[2].value)
                self.outgoing_points()
                self.incoming_points()
                self._lock.acquire()
                with self._count[0].get_lock():
                    self._count[0].value += 1
                # print('Lock acquired: {}/{}'.format(self._count[0].value,
                #                                     self._num_proc),
                #       self._count[1].value)
                if self._count[0].value < self._num_proc:
                    self._lock.wait()
                else:
                    with self._count[1].get_lock():
                        if self._count[1].value > 0:
                            with self._count[0].get_lock():
                                self._count[0].value = 0
                            self._count[1].value = 0
                        else:
                            with self._count[0].get_lock():
                                self._count[0].value = 0
                            self._count[1].value = 0
                            with self._count[2].get_lock():
                                self._count[2].value = 1
                    self._lock.notify_all()
                self._lock.release()
                # print 'Lock released', self._proc_idx,self._count[2].value
            if self._task == 'triangulate':
                self.enqueue_triangulation()
            elif self._task == 'volumes':
                self.enqueue_volumes()
            elif self._task == 'output':
                self.output_tess()
        # Synchronize to ensure rapid receipt
        self._lock.acquire()
        with self._count[0].get_lock():
            self._count[0].value += 1
        if self._count[0].value < self._num_proc:
            self._lock.wait()
        else:
            self._lock.notify_all()
        self._lock.release()
        self._done = True


class ParallelLeaf(object):
    r"""Wraps triangulation of a single leaf in a domain decomposition.

    Args:
        leaf (object): Leaf object from a tree returned by
            :meth:`cgal4py.domain_decomp.tree`.
        left_edges (np.ndarray): Minimums of each leaf in the domain
            decomposition.
        right_edges (np.ndarray): Maximums of each leaf in the domain
            decomposition.
        unique_str (str, optional): Unique string identifying the domain
            decomposition that will be used to construct an output file name.
            Default to None.

    Attributes:
        norig (int): The number of points originally located on this leaf.
        T (:class:`cgal4py.delaunay.Delaunay2` or
            :class:`cgal4py.delaunay.Delaunay3`:): 2D or 3D triangulation
            object.
        idx (np.ndarray of uint64): Indices of points on this leaf in the
            domain sorted position array (including those points transfered
            from other leaves).
        all_neighbors (set): Indices of all leaves that have been considered
            during particle exchanges.
        neighbors (list): Neighboring leaves that will be considered during the
            next particle exchange.
        left_neighbors (list): Neighboring leaves to the left of this leaf in
            each dimension.
        right_neighbors (list): Neighboring leaves to the right of this leaf in
            each dimension.
        left_edges (np.ndarray): Minimums of the domains in each dimension for
            leaves in `self.neighbors`.
        right_edges (np.ndarray): Maximums of the domains in each dimension for
            leaves in `self.neighbors`.
        unique_str (str): Unique string identifying the domain decomposition
            that will be used to construct an output file name.
        All attributes of `leaf`'s class also apply.

    """

    def __init__(self, leaf, left_edges, right_edges, unique_str=None):
        self._leaf = leaf
        self.norig = leaf.npts
        self.T = None
        if 10*leaf.stop_idx >= np.iinfo('uint32').max:
            self.idx = np.arange(leaf.start_idx,
                                 leaf.stop_idx).astype('uint64')
        else:
            self.idx = np.arange(leaf.start_idx,
                                 leaf.stop_idx).astype('uint32')
        self.all_neighbors = set([])
        self.neighbors = copy.deepcopy(leaf.neighbors)
        keep = False
        for i in xrange(self.ndim):
            if leaf.id in leaf.left_neighbors[i]:
                keep = True
                break
            if leaf.id in leaf.right_neighbors[i]:
                keep = True
                break
        if not keep:
            self.neighbors.remove(leaf.id)
        self.left_neighbors = copy.deepcopy(leaf.left_neighbors)
        self.right_neighbors = copy.deepcopy(leaf.right_neighbors)
        le = copy.deepcopy(left_edges)
        re = copy.deepcopy(right_edges)
        for i in xrange(self.ndim):
            if self.periodic_left[i]:
                for k in leaf.left_neighbors:
                    le[k, i] -= self.domain_width
                    re[k, i] -= self.domain_width
            if self.periodic_right[i]:
                for k in leaf.right_neighbors:
                    le[k, i] += self.domain_width
                    re[k, i] += self.domain_width
        self.left_edges = le[self.neighbors, :]
        self.right_edges = re[self.neighbors, :]
        self.unique_str = unique_str

    def __getattr__(self, name):
        if name in dir(self._leaf):
            return getattr(self._leaf, name)
        else:
            raise AttributeError

    def tessellate(self, pts):
        r"""Perform tessellation on leaf.

        Args:
            pts (np.ndarray of float64): (n,m) array of n m-dimensional
                coordinates.

        """
        self.T = Delaunay(pts)

    def outgoing_points(self):
        r"""Get indices of points that should be sent to each neighbor."""
        n = self.neighbors
        le = self.left_edges
        re = self.right_edges
        idx_enq = self.T.outgoing_points(le, re)
        # Remove points that are not local
        for i in xrange(len(n)):
            ridx = (idx_enq[i] < self.norig)
            idx_enq[i] = idx_enq[i][ridx]
        # Translate and add entries for non-neighbors
        hvall = [None for k in xrange(self.num_leaves)]
        for i, k in enumerate(n):
            hvall[k] = self.idx[idx_enq[i]]
        # Reset neighbors for incoming
        self.all_neighbors.update(self.neighbors)
        self.neighbors = []
        self.left_edges = np.zeros((0, self.ndim), 'float64')
        self.right_edges = np.zeros((0, self.ndim), 'float64')
        return hvall, n, le, re

    def outgoing_points_boundary(self):
        r"""Get indices of points that should be sent to each neighbor."""
        # TODO: Check that iind does not matter. iind contains points in tets
        # that are infinite. For non-periodic boundary conditions, these points
        # may need to be sent to distant leaves for an accurate convex hull.
        # Currently points on an edge without a bordering leaf are sent to all
        # leaves, but it is possible this could miss a few points...
        lind, rind, iind = self.T.boundary_points(self.left_edge,
                                                  self.right_edge,
                                                  True)
        # Remove points that are not local
        for i in xrange(self.ndim):
            ridx = (rind[i] < self.norig)
            lidx = (lind[i] < self.norig)
            rind[i] = rind[i][ridx]
            lind[i] = lind[i][lidx]
        # Count for preallocation
        all_leaves = range(0, self.id) + range(self.id + 1, self.num_leaves)
        Nind = np.zeros(self.num_leaves, 'uint32')
        for i in xrange(self.ndim):
            l_neighbors = self.left_neighbors[i]
            r_neighbors = self.right_neighbors[i]
            if len(l_neighbors) == 0:
                l_neighbors = all_leaves
            if len(r_neighbors) == 0:
                r_neighbors = all_leaves
            Nind[np.array(l_neighbors, 'uint32')] += len(lind[i])
            Nind[np.array(r_neighbors, 'uint32')] += len(rind[i])
        # Add points
        ln_out = [[[] for _ in xrange(self.ndim)] for
                  k in xrange(self.num_leaves)]
        rn_out = [[[] for _ in xrange(self.ndim)] for
                  k in xrange(self.num_leaves)]
        hvall = [np.zeros(Nind[k], rind[0].dtype) for
                 k in xrange(self.num_leaves)]
        Cind = np.zeros(self.num_leaves, 'uint32')
        for i in range(self.ndim):
            l_neighbors = self.left_neighbors[i]
            r_neighbors = self.right_neighbors[i]
            if len(l_neighbors) == 0:
                l_neighbors = all_leaves
            if len(r_neighbors) == 0:
                r_neighbors = all_leaves
            ilN = len(lind[i])
            irN = len(rind[i])
            for k in l_neighbors:
                hvall[k][Cind[k]:(Cind[k]+ilN)] = lind[i]
                Cind[k] += ilN
                for j in range(0, i) + range(i + 1, self.ndim):
                    rn_out[k][i] += self._leaf.left_neighbors[j]
                for j in range(self.ndim):
                    rn_out[k][i] += self._leaf.right_neighbors[j]
            for k in r_neighbors:
                hvall[k][Cind[k]:(Cind[k]+irN)] = rind[i]
                Cind[k] += irN
                for j in range(0, i) + range(i + 1, self.ndim):
                    ln_out[k][i] += self._leaf.right_neighbors[j]
                for j in range(self.ndim):
                    ln_out[k][i] += self._leaf.left_neighbors[j]
        # Ensure unique values (overlap can happen if a point is at a corner)
        for k in xrange(self.num_leaves):
            hvall[k] = self.idx[np.unique(hvall[k])]
            for i in range(self.ndim):
                ln_out[k][i] = list(set(ln_out[k][i]))
                rn_out[k][i] = list(set(rn_out[k][i]))
        return hvall, ln_out, rn_out

    def incoming_points(self, leafid, idx, n, le, re, pos):
        r"""Add incoming points from other leaves.

        Args:
            leafid (int): ID for the leaf that points came from.
            idx (np.ndarray of int): Indices of points being recieved.
            n (list of int): Indices of new neighbor leaves to add.
            le (np.ndarray of float64): Mins of new neighbor leaves in each
                dimension.
            re (np.ndarray of float64): Maxes of new neighbor leaves in each
                dimension.
            pos (np.ndarray of float): Positions of points being recieved.

        """
        if idx is None or idx.shape[0] == 0:
            return
        # Wrap points
        if self.id == leafid:
            for i in range(self.ndim):
                if self.periodic_left[i] and self.periodic_right[i]:
                    idx_left = ((pos[:, i] - self.left_edge[i]) <
                                (self.right_edge[i] - pos[:, i]))
                    idx_right = ((self.right_edge[i] - pos[:, i]) <
                                 (pos[:, i] - self.left_edge[i]))
                    pos[idx_left, i] += self.domain_width[i]
                    pos[idx_right, i] -= self.domain_width[i]
        else:
            for i in range(self.ndim):
                if self.periodic_right[i] and leafid in self.right_neighbors:
                    idx_left = ((pos[:, i] + self.domain_width[i] -
                                 self.right_edge[i]) <
                                (self.left_edge[i] - pos[:, i]))
                    pos[idx_left, i] += self.domain_width[i]
                if self.periodic_left[i] and leafid in self.left_neighbors:
                    idx_right = ((self.left_edge[i] - pos[:, i] +
                                  self.domain_width[i]) <
                                 (pos[:, i] - self.right_edge[i]))
                    pos[idx_right, i] -= self.domain_width[i]
        # Concatenate arrays
        self.idx = np.concatenate([self.idx, idx])
        # Insert points
        self.T.insert(pos)
        # Add neighbors
        for i in range(len(n)):
            if (n[i] != self.id) and (n[i] not in self.all_neighbors):
                self.neighbors.append(n[i])
                self.left_edges = np.vstack([self.left_edges, le[i, :]])
                self.right_edges = np.vstack([self.right_edges, re[i, :]])

    def incoming_points_boundary(self, leafid, idx, ln, rn, pos):
        r"""Add incoming points from other leaves.

        Args:
            leafid (int): ID for the leaf that points came from.
            idx (np.ndarray of int): Indices of points being recieved.
            rn (list of lists): Right neighbors that should be added in each
                dimension.
            ln (list of lists): Left neighbors that should be added in each
                dimension.
            pos (np.ndarray of float): Positions of points being recieved.

        """
        if idx.shape[0] == 0:
            return
        # Wrap points
        if self.id == leafid:
            for i in range(self.ndim):
                if self.periodic_left[i] and self.periodic_right[i]:
                    idx_left = ((pos[:, i] - self.left_edge[i]) <
                                (self.right_edge[i] - pos[:, i]))
                    idx_right = ((self.right_edge[i] - pos[:, i]) <
                                 (pos[:, i] - self.left_edge[i]))
                    pos[idx_left, i] += self.domain_width[i]
                    pos[idx_right, i] -= self.domain_width[i]
        else:
            for i in range(self.ndim):
                if self.periodic_right[i] and leafid in self.right_neighbors:
                    idx_left = ((pos[:, i] + self.domain_width[i] -
                                 self.right_edge[i]) <
                                (self.left_edge[i] - pos[:, i]))
                    pos[idx_left, i] += self.domain_width[i]
                if self.periodic_left[i] and leafid in self.left_neighbors:
                    idx_right = ((self.left_edge[i] - pos[:, i] +
                                  self.domain_width[i]) <
                                 (pos[:, i] - self.right_edge[i]))
                    pos[idx_right, i] -= self.domain_width[i]
        # Concatenate arrays
        self.idx = np.concatenate([self.idx, idx])
        # Insert points
        self.T.insert(pos)
        # Add neighbors
        for i in range(self.ndim):
            if self.id in ln[i]:
                ln[i].remove(self.id)
            if self.id in rn[i]:
                rn[i].remove(self.id)
            self.left_neighbors[i] = ln[i]
            self.right_neighbors[i] = rn[i]

    @property
    def tess_output_filename(self):
        r"""The default filename that should be used for tessellation
        output."""
        return _leaf_tess_filename(self.id, unique_str=self.unique_str)

    # def consolidate(self, ncells, idx_inf, all_verts, all_cells,
    #                 leaf_start, leaf_stop, split_map, inf_map):
    #     r"""Add local tessellation to global one.

    #     Args:
    #         ncells: Total number of cells currently in the global
    #             tessellation.
    #         split_map: Tuple containing necessary arrays to reconstruct the
    #             map containing information for cells split between leaves.
    #         inf_map: Tuple containing necessary arrays to reconstruct the map
    #             containing information for cells that are infinite.

    #     Returns:
    #         ncells: Total number of cells in the global tessellation after
    #             adding this leaf.
    #         split_map: Tuple containing necessary arrays to reconstruct the
    #             map containing information for cells split between leaves,
    #             updated after adding this leaf.
    #         inf_map: Tuple containing necessary arrays to reconstruct the map
    #             containing information for cells that are infinite, updated
    #             after adding this leaf.

    #     """
    #     ncells, split_map, inf_map = tools.add_leaf(
    #         self.ndim, ncells, idx_inf, all_verts, all_cells,
    #         leaf_start, leaf_stop,
    #         split_map[0], split_map[1], inf_map[0], inf_map[1],
    #         leaf.id, leaf.idx_inf, leaf.verts, leaf.neigh,
    #         leaf.sort_verts, leaf.sort_cells)
    #     return ncells, split_map, inf_map

    def serialize(self, store=False):
        r"""Get the serialized tessellation for this leaf.

        Args:
            store (bool, optional): If True, values are stored as attributes
                and not returned. Defaults to False.

        Returns:
            tuple: Vertices and neighbors for cells in the triangulation.

        """
        cells, neigh, idx_inf = self.T.serialize_info2idx(self.norig, self.idx)
        idx_verts, idx_cells = tools.py_arg_sortSerializedTess(cells)
        if store:
            self.idx_inf = idx_inf
            self.verts = cells
            self.neigh = neigh
            self.sort_verts = idx_verts
            self.sort_cells = idx_cells
        else:
            ncell_tot = self.T.num_cells
            return cells, neigh, idx_inf, idx_verts, idx_cells, ncell_tot

    def write_tess_to_file(self, fname=None):
        r"""Write out serialized information about the tessellation on this
        leaf.

        Args:
            fname (str, optional): Full path to file where tessellation info
                should be written. Defaults to None. If None, it is set to
                :method:`cgal4py.parallel.ParallelLeaf.tess_output_filename`.

        Returns:
            int: The maximum number of cells that will be contributed by this
                leaf. This is based on the number of cells found to be on this
                leaf in the local tessellation and includes cells that are not
                output to file (e.g. infinite cells).

        """
        if fname is None:
            fname = self.tess_output_filename
        out = self.serialize()
        cells, neigh, idx_inf, idx_verts, idx_cells, ncell_tot = out
        tools.output_leaf(fname, self.id, idx_inf, cells, neigh,
                          idx_verts, idx_cells, self.start_idx, self.stop_idx)
        return ncell_tot

    def voronoi_volumes(self):
        r"""Get the voronoi cell volumes for the original vertices on this
        leaf.

        Returns:
            np.ndarray of float64: Voronoi cell volumes. -1 indicates an
                infinite cell.

        """
        return self.T.voronoi_volumes()[:self.norig]
