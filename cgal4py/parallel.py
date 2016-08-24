from cgal4py.domain_decomp import Leaf
from cgal4py.delaunay import Delaunay

import numpy as np
import sys, os, time

import multiprocessing as mp

def ParallelDelaunay(pts, leaves, nproc, use_double=False):
    r"""Return a triangulation that is constructed in parallel.

    Args:
        leaves (list of :class:`cgal4py.domain_decomp.Leaf`): Domain 
            decomposition leaves that should be split amoung the processes.
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates.
        nproc (int): Number of processors that should be used.
        use_double (bool, optional): If True, the triangulation is forced to use 
            64bit integers reguardless of if there are too many points for 32bit. 
            Otherwise 32bit integers are used so long as the number of points is 
            <=4294967295. Defaults to False.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:
            consolidated 2D or 3D triangulation object.

    """
    # mpi4py version
    # script = os.path.join(os.path.dirname(__file__),'scripts','worker_leaf.py')
    # comm = MPI.COMM_SELF.Spawn(sys.executable, args=[script], maxprocs=nproc)
    # leaves = comm.scatter(task2leaves, root=MPI.ROOT)
    #
    # Split leaves
    task2leaves = [[] for _ in xrange(nproc)]
    for leaf in leaves:
        task = leaf.id % nproc
        task2leaves[task].append(leaf)
    # Create & execute processes
    queues = [mp.Queue() for _ in xrange(nproc)]
    processes = [DelaunayProcess(task2leaves[_], pts, queues, _) for _ in xrange(nproc)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    # Get leaves with tessellation
    tess_leaves = []
    for q in queues:
        tess_leaves += q.get()
        q.close()
    assert(len(leaves) == len(tess_leaves))
    # Consolidate tessellation
    return consolidate_leaves(tess_leaves, pts, use_double=use_double)

def parallelize_leaf(leaf, pts):
    r"""Convert :class:`cgal4py.domain_decomp.Leaf` into :class:`cgal4py.parallel.ParallelLeaf` 
    if it is not already parallelized by triangulating those points in the leaf.

    Args:
        leaf (:class:`cgal4py.domain_decomp.Leaf` or :class:`cgal4py.parallel.ParallelLeaf`):
            Leaf that should be parallelized.
        pts (np.ndarray of float64): (n,m) Array of n mD points. 

    Raises:
        ValueError: If `leaf` is not of class :class:`cgal4py.domain_decomp.Leaf` 
            or :class:`cgal4py.parallel.ParallelLeaf`.  

    """
    if isinstance(leaf, ParallelLeaf):
        pass
    elif isinstance(leaf, Leaf):
        leaf.__class__ = ParallelLeaf
        leaf.T = Delaunay(pts[leaf.idx,:])
        leaf.wrapped = np.zeros(len(leaf.idx), 'bool')
    else:
        raise ValueError("Only a cgal4py.domain_decomp.Leaf object can be turned into a ParallelLeaf.")

def consolidate_leaves(leaves, pts, use_double=False):
    r"""Creates a single triangulation from the triangulations of leaves.

    Args:
        leaves (list of :class:`cgal4py.parallel.ParallelLeaf`): Leaves from a 
            domain decomposition with triangulation that include points from 
            neighboring leaves.
        pts (np.ndarray of float64): (n,m) Array of n mD points. 
        use_double (bool, optional): If True, the triangulation is forced to use 
            64bit integers reguardless of if there are too many points for 32bit. 
            Otherwise 32bit integers are used so long as the number of points is 
            <=4294967295. Defaults to False.

    Returns:
        :class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:
            consolidated 2D or 3D triangulation object.

    """
    if pts.shape[0] >= np.iinfo('uint32').max:
        use_double = True
    T = Delaunay(np.zeros([0,pts.shape[1]]), use_double=use_double)
    return T

class ParallelLeaf(Leaf):
    r"""Wraps triangulation of a single leaf in a domain decomposition. For 
    additional information, see :class:`cgal4py.domain_decomp.Leaf`.

    Attributes: 
        T (:class:`cgal4py.delaunay.Delaunay2` or :class:`cgal4py.delaunay.Delaunay3`:): 
            2D or 3D triangulation object.
        All attributes of :class:`cgal4py.domain_decomp.Leaf` also apply.

    """
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
        # Count for preallocation
        Nind = np.zeros(self.num_leaves, 'uint32')
        for i in range(self.ndim):
            l_neighbors = list(set(self.neighbors[i]['left'] + \
              self.neighbors[i]['left_periodic']))
            r_neighbors = list(set(self.neighbors[i]['right'] + \
              self.neighbors[i]['right_periodic']))
            if len(l_neighbors) == 0:
                l_neighbors = range(self.num_leaves)
            if len(r_neighbors) == 0:
                r_neighbors = range(self.num_leaves)
            Nind[np.array(l_neighbors,'uint32')] += len(lind[i])
            Nind[np.array(r_neighbors,'uint32')] += len(rind[i])
        # Add points 
        hvall = [np.zeros(Nind[k], iind.dtype) for k in xrange(self.num_leaves)]
        Cind = np.zeros(self.num_leaves, 'uint32')
        for i in range(self.ndim):
            l_neighbors = list(set(self.neighbors[i]['left'] + \
              self.neighbors[i]['left_periodic']))
            r_neighbors = list(set(self.neighbors[i]['right'] + \
              self.neighbors[i]['right_periodic']))
            if len(l_neighbors) == 0:
                l_neighbors = range(self.num_leaves)
            if len(r_neighbors) == 0:
                r_neighbors = range(self.num_leaves)
            ilN = len(lind[i])
            irN = len(rind[i])
            for k in l_neighbors:
                hvall[k][Cind[k]:(Cind[k]+ilN)] = lind[i]
                Cind[k] += ilN
            for k in r_neighbors:
                hvall[k][Cind[k]:(Cind[k]+irN)] = rind[i]
                Cind[k] += irN
        # Ensure unique values (overlap can happen if a point is at a corner) 
        for k in xrange(self.num_leaves):
            hvall[k] = self.idx[np.unique(hvall[k])]
        return hvall

    def incoming_points(self, leafid, idx, pos):
        r"""Add incoming points from other leaves. 

        Args: 
            leafid (int): ID for the leaf that points came from. 
            idx (np.ndarray of int): Indices of points being recieved. 
            pos (np.ndarray of float): Positions of points being recieved. 

        """
        if len(idx) == 0: return
        # Wrap points
        wrapped = np.zeros(len(idx), 'bool')
        if self.id == leafid:
            for i in range(self.ndim):
                if self.periodic_left[i] and self.periodic_right[i]:
                    idx_left = (pos[:,i] - self.left_edge[i]) < (self.right_edge[i] - pos[:,i])
                    idx_right = (self.right_edge[i] - pos[:,i]) < (pos[:,i] - self.left_edge[i])
                    pos[idx_left,i] += self.domain_width[i]
                    pos[idx_right,i] -= self.domain_width[i]
                    wrapped[idx_left] = True
                    wrapped[idx_right] = True
        else:
            for i in range(self.ndim):
                if leafid in self.neighbors[i]['right_periodic']:
                    idx_left = (pos[:,i] + self.domain_width[i] - self.right_edge[i]) < (self.left_edge[i] - pos[:,i]) 
                    pos[idx_left,i] += self.domain_width[i]
                    wrapped[idx_left] = True
                if leafid in self.neighbors[i]['left_periodic']:
                    idx_right = (self.left_edge[i] - pos[:,i] + self.domain_width[i]) < (pos[:,i] - self.right_edge[i])
                    pos[idx_right,i] -= self.domain_width[i]
                    wrapped[idx_right] = True
        # Concatenate arrays
        self.idx = np.concatenate([self.idx, idx])
        # self.pos = np.concatenate([self.pos, pos]) 
        self.wrapped = np.concatenate([self.wrapped, wrapped])
        # Insert points 
        self.T.insert(pos)


class DelaunayProcess(mp.Process):
    r"""`multiprocessing.Process` subclass for coordinating operations on a 
    single process during a parallel Delaunay triangulation.

    Args:
        leaves (list of :class:`cgal4py.domain_decomp.Leaf`): Leaves that should 
            be triangulated on this process.
        pts (np.ndarray of float64): (n,m) array of n m-dimensional coordinates. 
            Each leaf has a set of indices identifying coordinates within `pts` 
            that belong to that leaf.
        queues (list of `multiprocessing.Queue`): List of queues for every 
            process being used in the triangulation.
        **kwargs: Variable keyword arguments are passed to `multiprocessing.Process`.

    """

    def __init__(self, leaves, pts, queues, proc_idx, **kwargs):
        super(DelaunayProcess, self).__init__(**kwargs)
        self._leaves = leaves
        self._pts = pts
        self._queues = queues
        self._num_proc = len(queues)
        self._local_leaves = len(leaves)
        if self._local_leaves == 0:
            self._total_leaves = 0
        else:
            self._total_leaves = leaves[0].num_leaves
        self._proc_idx = proc_idx

    def tessellate_leaves(self):
        r"""Performs the tessellation for each leaf on this process."""
        for leaf in self._leaves:
            parallelize_leaf(leaf, self._pts)

    def outgoing_points(self):
        r"""Enqueues points at edges of each leaf's boundaries."""
        for leaf in self._leaves:
            hvall = leaf.outgoing_points()
            for i in xrange(self._total_leaves):
                task = i % self._num_proc
                self._queues[task].put((i,leaf.id,hvall[i]))

    def incoming_points(self):
        r"""Takes points from the queue and adds them to the triangulation."""
        queue = self._queues[self._proc_idx]
        count = 0
        while count < (self._local_leaves*self._total_leaves):
            count += 1
            i,j,arr = queue.get()
            if len(arr) == 0:
                continue
            # Find leaf this should go to
            for leaf in self._leaves:
                if leaf.id == i:
                    break
            # Add points to leaves
            leaf.incoming_points(j, arr, self._pts[arr,:])

    def finalize_process(self):
        r"""Enqueue resulting tessellation."""
        self._queues[self._proc_idx].put(self._leaves)
        
    def run(self):
        r"""Performs tessellation and communication for each leaf on this process."""
        self.tessellate_leaves()
        self.outgoing_points()
        self.incoming_points()
        self.finalize_process()
