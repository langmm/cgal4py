from cgal4py.domain_decomp import Leaf
from cgal4py.delaunay import Delaunay

import numpy as np
import sys, os, time, copy

import multiprocessing as mp

def ParallelDelaunay(pts, tree, nproc, use_double=False):
    r"""Return a triangulation that is constructed in parallel.

    Args:
        tree (object): Domain decomposition tree for splitting points among the 
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
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
    pts = pts[tree.idx, :]
    # Split leaves
    task2leaves = [[] for _ in xrange(nproc)]
    for leaf in tree.leaves:
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
    total_idx = [None for _ in xrange(tree.num_leaves)]
    serial = [None for _ in xrange(tree.num_leaves)]
    for q in queues:
        ileaves = q.get()
        for iid, iidx, s in ileaves:
            assert(tree.leaves[iid].id == iid)
            total_idx[iid] = iidx
            serial[iid] = s
        q.close()
    # Consolidate tessellation
    return consolidate_leaves(tree, total_idx, serial, pts, use_double=use_double)

def consolidate_leaves(tree, total_idx, serial, pts, use_double=False):
    r"""Creates a single triangulation from the triangulations of leaves.

    Args:
        tree (object): Domain decomposition tree for splitting points among the 
            processes. Produced by :meth:`cgal4py.domain_decomp.tree`.
        total_idx (list): Indices of points on each leaf at the end.
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
    if npts >= np.iinfo('uint32').max:
        use_double = True
    if use_double:
        new_idx_inf = np.iinfo('uint64').max
    else:
        new_idx_inf = np.iinfo('uint32').max
    T = Delaunay(np.zeros([0,ndim]), use_double=use_double)
    # Create empty arrays
    ncells = 0
    for s in serial:
        ncells += s[0].shape[0]
    # Incrementally add cells
    cells = np.zeros((ncells, ndim+1), int) - 1
    split_cells = {}
    ncells = 0
    incl_cells = []
    for leaf in tree.leaves:
        s = serial[leaf.id]
        idx = total_idx[leaf.id]
        icells = s[0]
        idx_inf = s[2]
        iincl = np.zeros(icells.shape[0], 'int') - 1
        for c in range(icells.shape[0]):
            verts = icells[c,:]
            finite = (verts != idx_inf)
            idx_verts = copy.copy(verts)
            idx_verts[finite] = idx[verts[finite]]
            idx_verts[np.logical_not(finite)] = new_idx_inf
            vmin = np.argmin(idx_verts)
            key_verts = tuple(sorted(idx_verts)) 
            # All points on this leaf
            if (min(idx_verts) >= leaf.start_idx) and \
              (max(idx_verts) < leaf.stop_idx):
                iincl[c] = ncells
                cells[ncells, :] = idx_verts
                ncells += 1
                print 'All in', leaf.start_idx, leaf.stop_idx, idx_verts
            elif max(idx_verts) == new_idx_inf:
                if (min(idx_verts[finite]) >= leaf.start_idx) and \
                  (max(idx_verts[finite]) < leaf.stop_idx):
                    iincl[c] = ncells
                    cells[ncells, :] = idx_verts
                    ncells += 1
                elif (min(idx_verts) >= leaf.start_idx) and \
                  (min(idx_verts) < leaf.stop_idx):
                    if key_verts not in split_cells:
                        iincl[c] = ncells
                        cells[ncells, :] = idx_verts
                        split_cells[key_verts] = ncells
                        ncells += 1
                else:
                    print leaf.start_idx, leaf.stop_idx, idx_verts
            elif (min(idx_verts) >= leaf.start_idx) and \
                  (min(idx_verts) < leaf.stop_idx):
                if key_verts not in split_cells:
                    iincl[c] = ncells
                    cells[ncells, :] = idx_verts
                    split_cells[key_verts] = ncells
                    ncells += 1
            # # if min(verts) < leaf.norig:
            # # if min(idx_verts) in leaf.idx[:leaf.norig]:
            # # if True:
            # if verts[vmin] < leaf.norig:
            #     # print leaf.id, key_verts, verts, verts[vmin], leaf.norig
            #     # if True:
            #     if key_verts not in split_cells:
            #         iincl[c] = ncells
            #         cells[ncells, :] = idx_verts
            #         # key_verts = tuple(sorted(idx_verts))
            #         # if key_verts in split_cells:
            #         #     print verts, leaf.norig, vmin, idx_verts
            #         # if max(verts[finite]) > leaf.norig:
            #         split_cells[key_verts] = ncells
            #         ncells += 1
            # else:
            #     print verts, leaf.norig, vmin, idx_verts 
        incl_cells.append(iincl)
    cells = cells[:ncells,:]
    # Neighbors
    neighbors = np.zeros((ncells, ndim+1), int) - 1
    nnbors = 0
    for leaf in tree.leaves:
        idx = total_idx[leaf.id]
        s = serial[leaf.id]
        iincl = incl_cells[leaf.id]
        icells = s[0]
        ineighbors = s[1]
        idx_inf = s[2]
        for c_local in range(ineighbors.shape[0]):
            # Get info for this cell or its replacement
            if iincl[c_local] >= 0:
                c_total = nnbors
                nlist_total = range(icells.shape[1])
                nnbors += 1
            else:
                idx_verts = copy.copy(icells[c_local,:])
                finite = (idx_verts != idx_inf)
                idx_verts[finite] = idx[idx_verts[finite]]
                idx_verts[np.logical_not(finite)] = new_idx_inf
                c_total = split_cells.get(tuple(sorted(idx_verts)),-1)
                if c_total == -1: continue
                nlist_total = [list(cells[c_total,:]).index(_) for _ in idx_verts]
            # Loop over neighbors for this cell
            for n_local, n_total in enumerate(nlist_total):
                if iincl[ineighbors[c_local, n_local]] >= 0:
                    oth_c = iincl[ineighbors[c_local, n_local]]
                else:
                    idx_verts = copy.copy(icells[ineighbors[c_local, n_local],:])
                    finite = (idx_verts != idx_inf)
                    idx_verts[finite] = idx[idx_verts[finite]]
                    idx_verts[np.logical_not(finite)] = new_idx_inf
                    idx_verts = list(idx_verts)
                    oth_c = split_cells.get(tuple(sorted(idx_verts)),-1)
                if oth_c >= 0:
                    if (neighbors[c_total, n_total] >= 0) and (neighbors[c_total, n_total]!=oth_c):
                        print neighbors[c_total,:], n_total, oth_c
                    neighbors[c_total, n_total] = oth_c
    assert(ncells == nnbors)
    print cells
    print neighbors
    print(ncells, nnbors)
    print(np.sum(cells < 0),np.sum(neighbors < 0))
    assert(np.sum(cells < 0) == 0)
    assert(np.sum(neighbors < 0) == 0)
    if use_double:
        cells = cells.astype('uint64')
        neighbors = neighbors.astype('uint64')
    else:
        cells = cells.astype('uint32')
        neighbors = neighbors.astype('uint32')
    # Deserialize
    T.deserialize(pts, cells, neighbors, new_idx_inf)
    return T

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
        self._leaves = [ParallelLeaf(leaf) for leaf in leaves]
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
            leaf.tessellate(self._pts)

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
        out = [(leaf.id,leaf.idx,leaf.T.serialize()) for leaf in self._leaves]
        self._queues[self._proc_idx].put(out)
        
    def run(self):
        r"""Performs tessellation and communication for each leaf on this process."""
        self.tessellate_leaves()
        self.outgoing_points()
        self.incoming_points()
        self.finalize_process()

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
        # self.wrapped = np.zeros(leaf.npts, 'bool')

    def __getattr__(self, name):
        if name in dir(self._leaf):
            return getattr(self._leaf, name)
        else:
            raise AttributeError

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
        hvall = [np.zeros(Nind[k], iind.dtype) for k in xrange(self.num_leaves)]
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



