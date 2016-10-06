from cgal4py.delaunay import Delaunay, tools

import numpy as np
import sys, os, time, copy

import multiprocessing as mp

def ParallelVoronoiVolumes(pts, tree, nproc, use_double=False):
    r"""Return the voronoi cell volumes after constructing triangulation in 
    parallel.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates.
        tree (object): Domain decomposition tree for splitting points among the 
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        nproc (int): Number of processors that should be used.
        use_double (bool, optional): If True, the triangulation is forced to use 
            64bit integers reguardless of if there are too many points for 32bit. 
            Otherwise 32bit integers are used so long as the number of points is 
            <=4294967295. Defaults to False.

    Returns:
        np.ndarray of float64: (n,) array of n voronoi volumes for the provided 
            points.

    """
    pts = pts[tree.idx, :]
    # Split leaves
    task2leaves = [[] for _ in xrange(nproc)]
    for leaf in tree.leaves:
        task = leaf.id % nproc
        task2leaves[task].append(leaf)
    # Create & execute processes
    count = [mp.Value('i',0),mp.Value('i',0),mp.Value('i')]
    lock = mp.Condition()
    queues = [mp.Queue() for _ in xrange(nproc+1)]
    processes = [DelaunayProcess('volumes',task2leaves[_], pts, queues, 
                                 lock, count, _) for _ in xrange(nproc)]
    for p in processes:
        p.start()
    # Get leaves with tessellation
    serial = [None for _ in xrange(tree.num_leaves)]
    count = 0
    while count < nproc:
        ileaves = queues[-1].get()
        for iid, s in ileaves:
            assert(tree.leaves[iid].id == iid)
            serial[iid] = s
        count += 1
        time.sleep(0.01)
    # Consolidate volumes
    vol = np.empty(pts.shape[0], pts.dtype)
    for i,leaf in enumerate(tree.leaves):
        vol[tree.idx[leaf.slice]] = serial[i]
    # Close queues and processes
    for q in queues:
        q.close()
    for p in processes:
        p.join()
        p.terminate()
    return vol

def ParallelDelaunay(pts, tree, nproc, use_double=False):
    r"""Return a triangulation that is constructed in parallel.

    Args:
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates.
        tree (object): Domain decomposition tree for splitting points among the 
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        nproc (int): Number of processors that should be used.
        use_double (bool, optional): If True, the triangulation is forced to use 
            64bit integers reguardless of if there are too many points for 32bit. 
            Otherwise 32bit integers are used so long as the number of points is 
            <=4294967295. Defaults to False.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:
            consolidated 2D or 3D triangulation object.

    """
    pts = pts[tree.idx, :]
    # Split leaves
    task2leaves = [[] for _ in xrange(nproc)]
    for leaf in tree.leaves:
        task = leaf.id % nproc
        task2leaves[task].append(leaf)
    # Create & execute processes
    count = [mp.Value('i',0),mp.Value('i',0),mp.Value('i',0)]
    lock = mp.Condition()
    queues = [mp.Queue() for _ in xrange(nproc+1)]
    processes = [DelaunayProcess('triangulate',task2leaves[_], pts, queues, 
                                 lock, count, _) for _ in xrange(nproc)]
    for p in processes:
        p.start()
    # Get leaves with tessellation
    serial = [None for _ in xrange(tree.num_leaves)]
    count = 0
    while count < nproc:
        ileaves = queues[-1].get()
        for iid, s in ileaves:
            assert(tree.leaves[iid].id == iid)
            serial[iid] = s
        count += 1
        time.sleep(0.01)
    # Consolidate tessellation
    t0 = time.time()
    out = consolidate_tess(tree, serial, pts, use_double=use_double)
    t1 = time.time()
    print("Consolidation took {} s".format(t1-t0))
    # Close queues and processes
    for q in queues:
        q.close()
    for p in processes:
        p.join()
    # for p in processes:
        p.terminate()
    return out

def consolidate_tess(tree, serial, pts, use_double=False):
    r"""Creates a single triangulation from the triangulations of leaves.

    Args:
        tree (object): Domain decomposition tree for splitting points among the 
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        serial (list): Serialized tessellation info for each leaf.
        pts (np.ndarray of float64): (n,m) Array of n mD points. 
        use_double (bool, optional): If True, the triangulation is forced to use 
            64bit integers reguardless of if there are too many points for 32bit. 
            Otherwise 32bit integers are used so long as the number of points is 
            <=4294967295. Defaults to False.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:
            consolidated 2D or 3D triangulation object.

    """
    npts = pts.shape[0]
    ndim = pts.shape[1]
    num_leaves = tree.num_leaves
    if npts >= np.iinfo('uint32').max:
        use_double = True
    if use_double:
        idx_inf = np.uint64(np.iinfo('uint64').max)
    else:
        idx_inf = np.uint32(np.iinfo('uint32').max)
    # Get starting/stoping index for original particles on each leaf
    leaf_vidx_start = np.empty(num_leaves, 'uint64')
    leaf_vidx_stop = np.empty(num_leaves, 'uint64')
    for i in xrange(num_leaves):
        leaf_vidx_start[i] = tree.leaves[i].start_idx
        leaf_vidx_stop[i] = tree.leaves[i].stop_idx
    # Consolidate leaves
    cells, neigh = tools.consolidate_leaves(ndim, idx_inf, serial,
                                            leaf_vidx_start, leaf_vidx_stop)
    ncells = cells.shape[0]
    # if np.sum(neigh == idx_inf) != 0:
    #     for i in xrange(ncells):
    #         print i,cells[i,:], neigh[i,:]
    assert(np.sum(neigh == idx_inf) == 0)
    # Translate vertices to original values in tree (move to cython?)
    for i in xrange(ncells):
        for j in range(ndim+1):
            if cells[i,j] != idx_inf:
                cells[i,j] = tree.idx[cells[i,j]]
    # if cells.dtype != type(idx_inf):
    #     cells = cells.astype(type(idx_inf))
    #     neigh = neigh.astype(type(idx_inf))
    # Do tessellation
    T = Delaunay(np.zeros([0,ndim]), use_double=use_double)
    T.deserialize(pts[tree.idx,:], cells, neigh, idx_inf)
    return T


class DelaunayProcess(mp.Process):
    r"""`multiprocessing.Process` subclass for coordinating operations on a 
    single process during a parallel Delaunay triangulation.

    Args:
        task (str): Key for the task that should be parallelized. Options are:
              'triangulate': Perform triangulation and put serialized info in 
                  the output queue.
              'volumes': Perform triangulation and put volumes in output queue.
        leaves (list of leaf objects): Leaves that should be triangulated on 
            this process. The leaves are created by :meth:`cgal4py.domain_decomp.tree`.
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates. 
            Each leaf has a set of indices identifying coordinates within `pts` 
            that belong to that leaf.
        queues (list of `multiprocessing.Queue`): List of queues for every 
            process being used in the triangulation plus one for the main 
            process.
        lock (multiprocessing.Lock): Lock for processes.
        count (multiprocessing.Value): Shared integer for tracking exchanged 
            points.
        proc_idx (int): Index of this process.
        **kwargs: Variable keyword arguments are passed to `multiprocessing.Process`.

    Raises:
        ValueError: if `task` is not one of the accepted values listed above.

    """

    def __init__(self, task, leaves, pts, queues, lock, count, proc_idx, **kwargs):
        if task not in ['tessellate','exchange','enqueue_tess','enqueue_vols','triangulate','volumes']:
            raise ValueError('{} is not a valid task.'.format(task))
        super(DelaunayProcess, self).__init__(**kwargs)
        self._task = task
        self._leaves = [ParallelLeaf(leaf) for leaf in leaves]
        self._pts = pts
        self._lock = lock
        self._queues = queues
        self._count = count
        self._num_proc = len(queues)-1
        self._local_leaves = len(leaves)
        if self._local_leaves == 0:
            self._total_leaves = 0
        else:
            self._total_leaves = leaves[0].num_leaves
        self._proc_idx = proc_idx
        self._done = False
        self._nrecv = 1

    # def plot_leaves(self, plotbase=None):
    #     r"""Plots the tessellation for each leaf on this process.

    #     Args:
    #         plotbase (str, optional): Base path to which leaf IDs should be 
    #             appended to create plot filenames. Defaults to None. If None, 
    #             set to `'Process{}of{}_'.format(self._proc_idx,self._num_proc)`.

    #     """
    #     if plotbase is None:
    #         plotbase = 'Process{}of{}_'.format(self._proc_idx,self._num_proc)
    #     for leaf in self._leaves:
    #         leaf.plot(plotbase=plotbase)

    def tessellate_leaves(self):
        r"""Performs the tessellation for each leaf on this process."""
        for leaf in self._leaves:
            leaf.tessellate(self._pts)

    def outgoing_points(self):
        r"""Enqueues points at edges of each leaf's boundaries."""
        for leaf in self._leaves:
            hvall, ln, rn = leaf.outgoing_points()
            for i in xrange(self._total_leaves):
                task = i % self._num_proc
                self._queues[task].put((i,leaf.id,hvall[i],ln[i],rn[i]))
                time.sleep(0.01)

    def incoming_points(self):
        r"""Takes points from the queue and adds them to the triangulation."""
        queue = self._queues[self._proc_idx]
        count = 0
        nrecv = 0
        while count < (self._local_leaves*self._total_leaves):
            count += 1
            i,j,arr,ln,rn = queue.get()
            if len(arr) == 0:
                continue
            # Find leaf this should go to
            for leaf in self._leaves:
                if leaf.id == i:
                    break
            # Add points to leaves
            leaf.incoming_points(j, arr, ln, rn, self._pts[arr,:])
            nrecv += len(arr)
        self._nrecv = nrecv
        with self._count[0].get_lock():
            self._count[0].value += 1
        with self._count[1].get_lock():
            self._count[1].value += nrecv

    def enqueue_triangulation(self):
        r"""Enqueue resulting tessellation."""
        out = [(leaf.id,leaf.serialize()) for leaf in self._leaves]
        self._queues[-1].put(out)

    def enqueue_volumes(self):
        r"""Enqueue resulting voronoi volumes."""
        out = [(leaf.id,leaf.voronoi_volumes()) for leaf in self._leaves]
        self._queues[-1].put(out)

    def run(self):
        r"""Performs tessellation and communication for each leaf on this process."""
        if self._task == 'tessellate':
            self.tessellate_leaves()
        elif self._task == 'exchange':
            self.outgoing_points()
            self.incoming_points()
        elif self._task == 'enqueue_tess':
            self.enqueue_triangulation()
        elif self._task == 'enqueue_vols':
            self.enqueue_volumes()
        elif self._task == 'triangulate':
            self.tessellate_leaves()
            self.outgoing_points()
            self.incoming_points()
            # while self._count[2].value == 0:
            #     # print 'Begin',self._proc_idx, self._count[0].value, self._count[1].value, self._count[2].value
            #     self.outgoing_points()
            #     self.incoming_points()
            #     self._lock.acquire()
            #     # print 'Lock acquired: {}/{}'.format(self._count[0].value,self._num_proc), self._count[1].value
            #     if self._count[0].value < self._num_proc:
            #         self._lock.wait()
            #         self._lock.release()
            #     else:
            #         if self._count[1].value > 0:
            #             self._count[0].value = 0
            #             self._count[1].value = 0
            #             # self._count[2].value = 1
            #         else:
            #             self._count[2].value = 1
            #         self._lock.notify_all()
            #         self._lock.release()
            #     # print 'Lock released', self._proc_idx,self._count[2].value
            self.enqueue_triangulation()
        elif self._task == 'volumes':
            self.tessellate_leaves()
            self.outgoing_points()
            self.incoming_points()
            self.enqueue_volumes()
        self._done = True

class ParallelLeaf(object):
    r"""Wraps triangulation of a single leaf in a domain decomposition. 

    Args:
        leaf (object): Leaf object from a tree returned by 
            :meth:`cgal4py.domain_decomp.tree`.

    Attributes: 
        norig (int): The number of points originally located on this leaf.
        T (:class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:): 
            2D or 3D triangulation object.
        idx (np.ndarray of uint64): Indices of points on this leaf in the domain 
            sorted position array (including those points transfered from other 
            leaves).
        All attributes of `leaf`'s class also apply.

    """

    def __init__(self, leaf):
        self._leaf = leaf
        self.norig = leaf.npts
        self.T = None
        self.idx = np.arange(leaf.start_idx, leaf.stop_idx)
        self.left_neighbors = copy.deepcopy(leaf.left_neighbors)
        self.right_neighbors = copy.deepcopy(leaf.right_neighbors)
        # self.wrapped = np.zeros(leaf.npts, 'bool')

    def __getattr__(self, name):
        if name in dir(self._leaf):
            return getattr(self._leaf, name)
        else:
            raise AttributeError

    # def plot(self, plotbase=''):
    #     r"""Plots the tessellation for this leaf.

    #     Args:
    #         plotbase (str, optional): Base path to which leaf ID should be 
    #             appended to create plot filenames. Defaults to empty string.

    #     """
    #     plotfile = plotbase+'leaf{}of{}'.format(self.id,self.num_leaves)
    #     self.T.plot(plotfile=plotfile)

    def tessellate(self, pts):
        r"""Perform tessellation on leaf.

        Args:
            pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates.

        """
        self.T = Delaunay(copy.copy(pts[self.slice,:]))

    def outgoing_points(self):
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
        for i in range(self.ndim):
            ridx = (rind[i] < self.norig)
            lidx = (lind[i] < self.norig)
            rind[i] = rind[i][ridx]
            lind[i] = lind[i][lidx]
        # Count for preallocation
        all_leaves = range(0,self.id)+range(self.id+1,self.num_leaves)
        Nind = np.zeros(self.num_leaves, 'uint32')
        for i in range(self.ndim):
            l_neighbors = self.left_neighbors[i]
            r_neighbors = self.right_neighbors[i]
            if len(l_neighbors) == 0:
                l_neighbors = all_leaves
            if len(r_neighbors) == 0:
                r_neighbors = all_leaves
            Nind[np.array(l_neighbors,'uint32')] += len(lind[i])
            Nind[np.array(r_neighbors,'uint32')] += len(rind[i])
        # Add points 
        ln_out = [[[] for _ in range(self.ndim)] for k in xrange(self.num_leaves)]
        rn_out = [[[] for _ in range(self.ndim)] for k in xrange(self.num_leaves)]
        hvall = [np.zeros(Nind[k], rind[0].dtype) for k in xrange(self.num_leaves)]
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
                for j in range(0,i)+range(i+1,self.ndim):
                    rn_out[k][i] += self._leaf.left_neighbors[j]
                for j in range(self.ndim):
                    rn_out[k][i] += self._leaf.right_neighbors[j]
            for k in r_neighbors:
                hvall[k][Cind[k]:(Cind[k]+irN)] = rind[i]
                Cind[k] += irN
                for j in range(0,i)+range(i+1,self.ndim):
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
        # return [[hvall[k], ln_out[k], rn_out[k]] for k in xrange(self.num_leaves)]

    def incoming_points(self, leafid, idx, ln, rn, pos):
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
        if len(idx) == 0: return
        # Wrap points
        # wrapped = np.zeros(len(idx), 'bool')
        if self.id == leafid:
            for i in range(self.ndim):
                if self.periodic_left[i] and self.periodic_right[i]:
                    idx_left = (pos[:,i] - self.left_edge[i]) < (self.right_edge[i] - pos[:,i])
                    idx_right = (self.right_edge[i] - pos[:,i]) < (pos[:,i] - self.left_edge[i])
                    pos[idx_left,i] += self.domain_width[i]
                    pos[idx_right,i] -= self.domain_width[i]
                    # wrapped[idx_left] = True
                    # wrapped[idx_right] = True
        else:
            for i in range(self.ndim):
                if self.periodic_right[i] and leafid in self.right_neighbors:
                    idx_left = (pos[:,i] + self.domain_width[i] - self.right_edge[i]) < (self.left_edge[i] - pos[:,i]) 
                    pos[idx_left,i] += self.domain_width[i]
                    # wrapped[idx_left] = True
                if self.periodic_left[i] and leafid in self.left_neighbors:
                    idx_right = (self.left_edge[i] - pos[:,i] + self.domain_width[i]) < (pos[:,i] - self.right_edge[i])
                    pos[idx_right,i] -= self.domain_width[i]
                    # wrapped[idx_right] = True
        # Concatenate arrays
        self.idx = np.concatenate([self.idx, idx])
        # self.pos = np.concatenate([self.pos, pos]) 
        # self.wrapped = np.concatenate([self.wrapped, wrapped])
        # Insert points 
        self.T.insert(pos)
        # Add neighbors
        for i in range(self.ndim):
            if self.id in ln[i]:
                ln[i].remove(self.id)
            if self.id in rn[i]:
                rn[i].remove(self.id)
            self.left_neighbors[i] = list(set(self.left_neighbors[i]+ln[i]))
            self.right_neighbors[i] = list(set(self.right_neighbors[i]+rn[i]))

    def serialize(self):
        r"""Get the serialized tessellation for this leaf.

        Returns:
            tuple: Vertices and neighbors for cells in the triangulation.

        """
        cells, neigh, idx_inf = self.T.serialize()
        for i in xrange(cells.shape[0]):
            for j in xrange(cells.shape[1]):
                if cells[i,j] != idx_inf:
                    cells[i,j] = self.idx[cells[i,j]]
        idx_verts, idx_cells = tools.py_arg_sortSerializedTess(cells)
        return cells, neigh, idx_inf, idx_verts, idx_cells

    def voronoi_volumes(self):
        r"""Get the voronoi cell volumes for the original vertices on this leaf.

        Returns:
            np.ndarray of float64: Voronoi cell volumes. -1 indicates an 
                infinite cell.

        """
        return self.T.voronoi_volumes()[:self.norig]

