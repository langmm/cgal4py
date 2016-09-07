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
    idx_sort = np.argsort(tree.idx)
    return consolidate_leaves(tree, total_idx, serial, pts[idx_sort, :], 
                                  idx_sort, use_double=use_double).T

class consolidate_leaves(object):

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
    def __init__(self, tree, total_idx, serial, pts, idx_sort, use_double=False):
        self.tree = tree
        self.num_leaves = len(tree.leaves)
        self.leaf_idx = total_idx
        self.leaf_cells = [s[0] for s in serial]
        self.leaf_neigh = [s[1] for s in serial]
        self.visited = [np.zeros(c.shape[0])-1 for c in self.leaf_cells]
        self.pts = pts
        self.npts = pts.shape[0]
        self.ndim = pts.shape[1]
        if self.npts >= np.iinfo('uint32').max:
            use_double = True
        if use_double:
            new_idx_inf = np.uint64(np.iinfo('uint64').max)
        else:
            new_idx_inf = np.uint32(np.iinfo('uint32').max)
        idx_inf = serial[0][2]
        for s in serial:
            assert(s[2] == idx_inf)
        self.use_double = use_double
        self.new_idx_inf = new_idx_inf
        self.idx_inf = idx_inf
        self.T = Delaunay(np.zeros([0,self.ndim]), use_double=use_double)
        ncells = 0
        for c in self.leaf_cells:
            ncells += c.shape[0]
        self.cells = np.zeros((ncells, self.ndim+1), int) - 1
        self.neigh = np.zeros((ncells, self.ndim+1), int) - 1
        self.ncells = 0
        self.nnbors = 0
        self.split_cells = {}
        for i in range(self.num_leaves):
            self.walk_cells(i)
        for i in range(self.num_leaves):
            self.walk_neigh(i)
        self.cells = self.cells[:self.ncells,:]
        self.neigh = self.neigh[:self.ncells,:]
        self.walk_final()
        # print self.cells
        # print self.neigh
        # print(np.sum(self.cells < 0),np.sum(self.neigh < 0))
        assert(np.sum(self.cells < 0) == 0)
        assert(np.sum(self.neigh < 0) == 0)
        cells = self.cells
        cells[cells != new_idx_inf] = idx_sort[cells[cells != new_idx_inf]]
        self.T.deserialize(pts, cells.astype(type(new_idx_inf)), 
                           self.neigh.astype(type(new_idx_inf)), new_idx_inf)

    def get_cell_verts(self, leafid, cidx):
        r"""Get the global leaf-sorted vertex indices for a cell on a leaf.

        Args:
            leafid (int): Index specifying a leaf in the tree.
            cidx (int): Index of a cell on the leaf.

        Returns:
            np.ndarray of int: Global leaf-sorted indices of vertices in the 
                specified cell.

        """
        verts = self.leaf_cells[leafid][cidx,:]
        finite = (verts != self.idx_inf)
        idx_verts = copy.copy(verts)
        idx_verts[finite] = self.leaf_idx[leafid][verts[finite]]
        idx_verts[np.logical_not(finite)] = self.new_idx_inf
        return idx_verts

    def get_cell_neigh(self, leafid, cidx):
        r"""Get the neighboring cells for a cell on a leaf.

        Args:
            leafid (int): Index specifying a leaf in the tree.
            cidx (int): Index of a cell on the leaf.

        Returns:
            np.ndarray of int: Indices of neighboring cells on this leaf.

        """
        return self.leaf_neigh[leafid][cidx,:]

    def add_cell(self, c, n, start=0, stop=None):
        r"""Add a cell to the serial tessellation.

        Args:
            c (np.ndarray of int): Indices of vertices in new cell.
            n (np.ndarray of int): Indices of neighbors to new cell.
            start (int, optional): Cell to start search for neighbors at. 
                Defaults to 0.
            stop (int, optional): Cell to stop search for neighbors at. 
                Defaults to None.

        """
        c, n = self.find_neigh(c, n, start=start, stop=stop)
        self.cells = np.concatenate([self.cells, c.reshape((1,len(c)))])
        self.neigh = np.concatenate([self.neigh, n.reshape((1,len(n)))])
        self.ncells += 1

    def find_leaves(self, c):
        r"""Determine which leaves originally owned vertices in the set.

        Args:
            c (np.ndarray of int): Vertex indices.

        Returns:
            list of int: Indices of leaves that own one or more of the vertices.

        """
        leaves = []
        for x in c:
            if x == self.new_idx_inf:
                pass
            for i in xrange(self.num_leaves):
                leaf = self.tree.leaves[i]
                if x < leaf.stop_idx:
                    leaves.append(i)
                    break
        return leaves

    def find_neigh(self, c1, n1, start=0, stop=None):
        r"""Look for neighbors in existing cells.

        Args:
            c1 (np.ndarray of int): Indices of vertices in new cell.
            n1 (np.ndarray of int): Indices of neighbors to new cell.
            start (int, optional): Cell to start search at. Defaults to 0.
            stop (int, optional): Cell to stop search at. Defaults to None. If 
                None, set to `self.ncells`.

        """
        if -1 not in n1:
            return c1, n1
        if stop is None:
            stop = self.ncells
        for i2 in range(start,stop):
            c2 = self.cells[i2,:]
            n2 = self.neigh[i2,:]
            matches1 = np.zeros(self.ndim+1, 'int') - 1
            matches2 = np.zeros(self.ndim+1, 'int') - 1
            for v2 in range(self.ndim+1):
                if c2[v2] in c1:
                    v1 = list(c1).index(c2[v2])
                    matches2[v2] = v1
                    matches1[v1] = v2
            if np.sum(matches1 < 0) == 1:
                n1[(matches1 < 0)] = i2
                self.neigh[i2,(matches2 < 0)] = self.ncells
        return c1, n1

    def walk_cells(self, curr_leaf):
        r"""Iterate over cells on a leaf, adding those to the global list that 
        have at least one vertex originating from that leaf.

        Args:
            curr_leaf (int): Index of the leaf that should be processed.

        """
        leaf = self.tree.leaves[curr_leaf]
        for curr_cell in range(self.leaf_cells[curr_leaf].shape[0]):
            verts = self.get_cell_verts(curr_leaf, curr_cell)
            neigh = self.get_cell_neigh(curr_leaf, curr_cell)
            fin = (verts != self.new_idx_inf)
            key = tuple(sorted(verts))
            # Infinite
            if (max(verts) == self.new_idx_inf):
                self.visited[curr_leaf][curr_cell] = -777
            # Finite
            else:
                leaves = self.find_leaves(verts)
                # All points on a single leaf
                if len(leaves) == 1:
                    # This leaf
                    if leaves[0] == curr_leaf:
                        self.cells[self.ncells,:] = verts
                        self.visited[curr_leaf][curr_cell] = self.ncells
                        self.ncells += 1
                    # Other leaf
                    else:
                        self.visited[curr_leaf][curr_cell] = -999
                # Split between leaves
                else:
                    if key in self.split_cells:
                        self.visited[curr_leaf][curr_cell] = -888
                    else:
                        self.cells[self.ncells,:] = verts
                        self.visited[curr_leaf][curr_cell] = self.ncells
                        self.split_cells[key] = self.ncells
                        self.ncells += 1

    def walk_neigh(self, curr_leaf):
        r"""Iterate over cells on a leaf, adding neighbors to the global list
        for those cells that were processed by :class:`consolidate_leaves.walk_cells`.

        Args:
            curr_leaf (int): Index of the leaf that should be processed.

        """
        leaf = self.tree.leaves[curr_leaf]
        for c_local in range(self.leaf_neigh[curr_leaf].shape[0]):
            neigh = self.get_cell_neigh(curr_leaf, c_local)
            # Get index for this cell or its replacement
            if self.visited[curr_leaf][c_local] >= 0:
                c_total = self.nnbors
                nlist_total = range(len(neigh))
                self.nnbors += 1
            elif self.visited[curr_leaf][c_local] == -888:
                verts = self.get_cell_verts(curr_leaf, c_local)
                c_total = self.split_cells[tuple(sorted(verts))]
                nlist_total = [list(self.cells[c_total,:]).index(_) for _ in verts]
            else:
                continue
            # Loop over neighbors for this cell  
            for n_local, n_total in enumerate(nlist_total):
                if self.visited[curr_leaf][neigh[n_local]] >= 0:
                    oth_c = self.visited[curr_leaf][neigh[n_local]]
                elif self.visited[curr_leaf][neigh[n_local]] == -888:
                    oth_c = self.split_cells[tuple(sorted(self.get_cell_verts(curr_leaf, neigh[n_local])))]
                else:
                    continue
                if (self.neigh[c_total, n_total] >= 0) and (self.neigh[c_total, n_total] != oth_c):
                    print(self.neigh[c_total,:], self.neigh[c_total, n_total], oth_c)
                else:
                    self.neigh[c_total, n_total] = oth_c

    def walk_final(self):
        r"""Perform a final iteration over the global cell neighbor list, adding 
        infinite cells where neighbors are missing."""
        norig = self.ncells
        mult_miss = []
        for c in range(norig):
            idx_neigh = (self.neigh[c,:] >= 0)
            Nneigh = np.sum(idx_neigh)
            if (Nneigh == self.ndim+1):
                continue
            elif (Nneigh == self.ndim):
                for idx_miss in np.where(self.neigh[c,:] < 0)[0]:
                    new_cell = np.zeros(self.ndim+1, 'int') - 1
                    new_cell = copy.deepcopy(self.cells[c,:])
                    new_cell[idx_miss] = self.new_idx_inf
                    new_neigh = np.zeros(self.ndim+1, 'int') - 1
                    new_neigh[idx_miss] = c
                    self.neigh[c, idx_miss] = self.ncells
                    self.add_cell(new_cell, new_neigh, start=norig, stop=self.ncells)
            else:
                print("{} neighbors missing from cell {}...".format(self.ndim+1-Nneigh,c))
                print("    cells = {}, neigh = {}".format(self.cells[c,:], self.neigh[c,:]))
                mult_miss.append(c)
        # Remove things
        # for r in sorted(remove)[::-1]:
        #     self.cells = np.delete(self.cells, r, axis=0)
        #     self.neigh = np.delete(self.neigh, r, axis=0)
        #     self.ncells -= 1
        # # for r in sorted(remove)[::-1]:
        # #     idx_r = (self.neigh == r)
        # #     assert(np.sum(idx_r) == 0)
        # # for r in sorted(remove)[::-1]:
        #     idx_upper = (self.neigh > r)
        #     self.neigh[idx_upper] -= 1


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



