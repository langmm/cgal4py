from cgal4py.domain_decomp import Leaf
from cgal4py.delaunay import Delaunay, tools

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
    queues = [mp.Queue() for _ in xrange(nproc+1)]
    processes = [DelaunayProcess(task2leaves[_], pts, queues, _) for _ in xrange(nproc)]
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
    for q in queues:
        q.close()
    for p in processes:
        p.join()
    # for p in processes:
        p.terminate()
    # Consolidate tessellation
    idx_sort = np.argsort(tree.idx)
    t0 = time.time()
    out = consolidate_leaves(tree, serial, pts,
                             idx_sort, use_double=use_double).T
    t1 = time.time()
    print("Consolidation took {} s".format(t1-t0))
    return out

class CellIndex(object):

    def __init__(self, ncell_max, ndim):
        self.ncell_max = ncell_max
        self.ndim = ndim
        self.count = 0
        self.index = np.zeros((ncell_max, ndim+1), 'int') - 1
        self.values = []

    def _find_cell(self, c):
        i = 0
        for d in range(self.ndim+1):
            while (i < self.count) and (c[d] > self.index[i,d]) and np.all(c[:d] >= self.index[i,:d]):
                i += 1
        if i >= self.count:
            equal = False
        else:
            equal = True
            for d in range(self.ndim+1):
                if self.index[i,d] != c[d]:
                    equal = False
                    break
        return c, i, equal

    def __contains__(self, c0):
        c, i, equal = self._find_cell(c0)
        return equal

    def __getitem__(self, c0):
        c, i, equal = self._find_cell(c0)
        if equal:
            return self.values[i]
        else:
            raise ValueError("The cell {} is not in the index.".format(c0))

    def get(self, c0, default=None):
        try:
            out = self[c0]
        except ValueError:
            if default is None:
                raise
            else:
                out = default
        return out

    def __setitem__(self, key, value):
        self.insert(key, value)

    def insert(self, c0, idx):
        if self.count >= self.ncell_max:
            raise RuntimeError("The cell count ({}) exceeded the max ({}).".format(self.count, self.ncell_max))
        c, i, equal = self._find_cell(c0)
        if equal:
            return self.values[i]
        else:
            self.index[(i+1):(self.count+1),:] = self.index[i:self.count,:]
            self.index[i,:(self.ndim+1)] = c
            self.values.insert(i, idx)
            self.count += 1
            return idx

class CellDict(object):

    def __init__(self):
        self.dict = {}

    def __contains__(self, c0):
        return tuple(c0) in self.dict

    def __getitem__(self, c):
        return self.dict[tuple(c0)]
    
    def get(self, default=None):
        return self.dict.get(tuple(c0), default)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def insert(self, c0, idx):
        c = tuple(c0)
        i = self.dict.get(c, idx)
        if i == idx:
            self.dict[c] = idx
            return idx
        else:
            return i

class consolidate_leaves(object):

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
    def __init__(self, tree, serial, pts, idx_sort, use_double=False):
        self.tree = tree
        self.num_leaves = len(tree.leaves)
        self.pts = pts
        self.npts = pts.shape[0]
        self.ndim = pts.shape[1]
        if self.npts >= np.iinfo('uint32').max:
            use_double = True
        if use_double:
            idx_inf = np.uint64(np.iinfo('uint64').max)
        else:
            idx_inf = np.uint32(np.iinfo('uint32').max)
        self.use_double = use_double
        self.idx_inf = idx_inf
        # Get counts
        self.leaf_counts = np.zeros(self.num_leaves,'int')
        self.leaf_start = np.zeros(self.num_leaves,'int')
        self.leaf_stop = np.zeros(self.num_leaves,'int')
        prev = 0
        for i,s in enumerate(serial):
            curr = s[0].shape[0]
            self.leaf_counts[i] = curr
            self.leaf_start[i] = prev
            self.leaf_stop[i] = prev + curr
            prev += curr
        self.ncells_orig = prev
        # Create master arrays
        self.leaf_cells = np.empty((self.ncells_orig, self.ndim+1), type(idx_inf))
        self.leaf_neigh = np.empty((self.ncells_orig, self.ndim+1), type(idx_inf))
        self.leaf_idx_verts = np.empty((self.ncells_orig, self.ndim+1), 'uint32')
        self.leaf_idx_cells = np.empty(self.ncells_orig, 'uint64')
        self.visited = np.zeros(self.ncells_orig, 'int') - 1
        for i,s in enumerate(serial):
            islc = slice(self.leaf_start[i], self.leaf_stop[i])
            self.leaf_cells[islc, :] = s[0]
            self.leaf_neigh[islc, :] = s[1]
            self.leaf_cells[islc, :][s[0] == s[2]] = self.idx_inf
            self.leaf_idx_verts[islc, :] = s[3]
            self.leaf_idx_cells[islc] = s[4] + self.leaf_start[i]
        self.T = Delaunay(np.zeros([0,self.ndim]), use_double=use_double)
        ncells = np.sum(self.leaf_counts)
        self.cells = np.zeros((ncells, self.ndim+1), int) - 1
        self.neigh = np.zeros((ncells, self.ndim+1), int) - 1
        self.source = np.zeros((ncells, 2), int) - 1
        self.ncells = 0
        self.split_dict = CellDict()
        self.split_index = CellIndex(ncells, self.ndim)
        self.inf_index = CellIndex(ncells, self.ndim)
        for i in range(self.num_leaves):
            self.walk_cells(i)
        self.cells = self.cells[:self.ncells,:]
        self.neigh = self.neigh[:self.ncells,:]
        self.source = self.source[:self.ncells,:]
        self.walk_inf()
        if np.sum(self.neigh < 0) != 0:
            for i in range(self.cells.shape[0]):
                print i,self.cells[i,:], self.neigh[i,:]
        assert(np.sum(self.cells < 0) == 0)
        assert(np.sum(self.neigh < 0) == 0)
        cells = self.cells
        cells[cells != idx_inf] = tree.idx[cells[cells != idx_inf]]
        self.T.deserialize(pts[tree.idx,:], cells.astype(type(idx_inf)), 
                           self.neigh.astype(type(idx_inf)), idx_inf)

    def get_cell_verts(self, leafid, cidx):
        r"""Get the global leaf-sorted vertex indices for a cell on a leaf.

        Args:
            leafid (int): Index specifying a leaf in the tree.
            cidx (int): Index of a cell on the leaf.

        Returns:
            np.ndarray of int: Global leaf-sorted indices of vertices in the 
                specified cell.

        """
        return self.leaf_cells[self.leaf_start[leafid]+cidx,:]

    def get_cell_neigh(self, leafid, cidx):
        r"""Get the neighboring cells for a cell on a leaf.

        Args:
            leafid (int): Index specifying a leaf in the tree.
            cidx (int): Index of a cell on the leaf.

        Returns:
            np.ndarray of int: Indices of neighboring cells on this leaf.

        """
        return self.leaf_neigh[self.leaf_start[leafid]+cidx,:]

    def get_cell_visit(self, leafid, cidx):
        r"""Get the status of the cell.

        Args:
            leafid (int): Index specifying a leaf in the tree.
            cidx (int): Index of a cell on the leaf.

        Returns:
            int: Integer specifying the status of the cell.

        """
        return self.visited[self.leaf_start[leafid]+int(cidx)]

    def set_cell_visit(self, leafid, cidx, value):
        r"""Set the status of the cell.

        Args:
            leafid (int): Index specifying a leaf in the tree.
            cidx (int): Index of a cell on the leaf.
            value (int): Specifier for cell status.

        """
        self.visited[self.leaf_start[leafid]+cidx] = value

    def walk_cells(self, curr_leaf):
        r"""Iterate over cells on a leaf, adding those to the global list that 
        have at least one vertex originating from that leaf.

        Args:
            curr_leaf (int): Index of the leaf that should be processed.

        """
        leaf = self.tree.leaves[curr_leaf]
        for curr_cell in xrange(self.leaf_counts[curr_leaf]):
            verts = self.get_cell_verts(curr_leaf, curr_cell)
            neigh = self.get_cell_neigh(curr_leaf, curr_cell)
            self.add_cell(curr_leaf, curr_cell, verts)
            self.add_neigh(curr_leaf, curr_cell, verts, neigh)

    def walk_inf(self):
        r"""Perform an iteration over the global cell neighbor list, adding 
        infinite cells where neighbors are missing."""
        norig = self.ncells
        mult_miss = []
        for c in xrange(norig):
            idx_neigh = (self.neigh[c,:] >= 0)
            Nneigh = np.sum(idx_neigh)
            if (Nneigh == self.ndim+1):
                continue
            elif (Nneigh <= self.ndim):
                for idx_miss in np.where(self.neigh[c,:] < 0)[0]:
                    new_cell = np.empty(self.ndim+1, 'int') - 1
                    idx_fwd = range(idx_miss) + range(idx_miss+1, self.ndim+1)
                    for i in range(self.ndim):
                        new_cell[idx_fwd[i]] = self.cells[c,idx_fwd[-(i+1)]]
                    new_cell[idx_miss] = self.idx_inf
                    new_neigh = np.zeros(self.ndim+1, 'int') - 1
                    new_neigh[idx_miss] = c
                    idx_sort_verts = np.argsort(new_cell)
                    idx = self.inf_index.insert(new_cell[idx_sort_verts], self.ncells)
                    self.neigh[c, idx_miss] = idx
                    if idx == self.ncells:
                        self.append_cell(new_cell, new_neigh, start=norig, stop=self.ncells)
            else:
                print("{} neighbors missing from cell {}...".format(self.ndim+1-Nneigh,c))
                print("    cells = {}, neigh = {}".format(self.cells[c,:], self.neigh[c,:]))
                src = self.source[c,:]
                print("    source = {}, visited = {}".format(src,self.get_cell_visit(src[0],src[1])))
                for n in self.get_cell_neigh(src[0], src[1]):
                    print('        ',self.get_cell_verts(src[0], n), self.get_cell_visit(src[0],n))
                mult_miss.append(c)

    def add_cell(self, curr_leaf, curr_cell, verts):
        if (self.get_cell_visit(curr_leaf,curr_cell) != -1):
            return
        idx_sort_verts = self.leaf_idx_verts[self.leaf_start[curr_leaf]+curr_cell,:]
        # Infinite
        if (verts[idx_sort_verts[0]] == self.idx_inf):
            self.set_cell_visit(curr_leaf,curr_cell,-777)
            return
        # Finite
        leaves = self.find_leaves(verts)
        # All points on a single leaf
        if len(leaves) == 1:
            # This leaf
            if leaves[0] == curr_leaf:
                idx = self.get_cell_visit(curr_leaf,curr_cell)
                if idx < 0:
                    idx = self.ncells
                    self.cells[idx,:] = verts
                    self.source[idx,:] = np.array([curr_leaf,curr_cell])
                    self.ncells += 1
                self.set_cell_visit(curr_leaf,curr_cell,idx)
            else:
                oth_leaf = leaves[0]
                oth_cell = self.find_cell_on_leaf(oth_leaf, verts, idx_sort_verts)
                if oth_cell >= 0:
                    idx = self.get_cell_visit(oth_leaf, oth_cell)
                    if idx < 0:
                        idx = self.ncells
                        self.cells[idx,:] = verts
                        self.source[idx,:] = np.array([curr_leaf,curr_cell])
                        self.set_cell_visit(oth_leaf,oth_cell,idx)
                        self.ncells += 1
                    self.set_cell_visit(curr_leaf,curr_cell,idx)
        # Split between leaves
        else:
            # idx = self.split_dict.insert(verts, self.ncells)
            idx = self.split_index.insert(verts[idx_sort_verts], self.ncells)
            self.set_cell_visit(curr_leaf,curr_cell,idx)
            if idx == self.ncells:
                self.cells[self.ncells,:] = verts
                self.source[self.ncells,:] = np.array([curr_leaf,curr_cell])
                self.ncells += 1

    def add_neigh(self, curr_leaf, curr_cell, verts, neigh):
        c_total = self.get_cell_visit(curr_leaf, curr_cell)
        if c_total < 0:
            return
        nlist_total = [list(self.cells[c_total,:]).index(_) for _ in verts]
        # Loop over neighbors for this cell  
        idx_sort = np.zeros(self.ndim+1, 'int') - 1
        new_neigh = self.neigh[c_total, :]
        for n_local in range(self.ndim+1):
            n_total = list(self.cells[c_total,:]).index(verts[n_local])
            assert(verts[n_local] == self.cells[c_total, n_total])
            oth_c = self.get_cell_visit(curr_leaf, neigh[n_local])
            if oth_c < 0:
                continue
            for n_other in range(self.ndim+1):
                if self.cells[oth_c, n_other] not in self.cells[c_total,:]:
                    break
            old_c = self.neigh[c_total, n_total]
            if (old_c >= 0): 
                if (old_c == oth_c):
                    # assert(self.neigh[old_c, n_total] == c_total)
                    # assert(n_total == n_other)
                    continue
                else:
                    print(n_total, n_other, self.cells[c_total,:], self.cells[old_c,:], self.cells[oth_c,:])
                    old_src = self.source[old_c,:]
                    oth_src = self.source[oth_c,:]
                    print("    old",old_src,self.get_cell_visit(old_src[0],old_src[1]))
                    print("    oth",oth_src,self.get_cell_visit(oth_src[0],oth_src[1]))
                    print("        leaves", self.find_leaves(self.cells[c_total,:],full=True),
                              self.find_leaves(self.cells[old_c,:],full=True), 
                              self.find_leaves(self.cells[oth_c,:],full=True))
            else:
                self.neigh[c_total, n_total] = oth_c
                self.neigh[oth_c, n_other] = c_total

    def find_swap(self, i, n1, n2, orig=None, level=0):
        if orig is None:
            orig = [i]
        else:
            if (level > 0) and (i in orig):
                return False
        if (n1 == n2):
            return True
        elif (self.neigh[i,n2] == -1):
            self.swap_neigh(i, n1, n2)
            return True
        else:
            out = self.find_swap(self.neigh[i,n2], n2, n1, orig=orig, level=level+1)
            if out:
                self.swap_neigh(i, n1, n2)
            return out
                
    def swap_neigh(self, i, n1, n2):
        tn = self.neigh[i, n1]
        tc = self.cells[i, n1]
        self.neigh[i, n1] = self.neigh[i, n2]
        self.cells[i, n1] = self.cells[i, n2]
        self.neigh[i, n2] = tn
        self.cells[i, n2] = tc

    def find_cell_on_leaf(self, curr_leaf, verts, idx_sort):
        i = self.leaf_start[curr_leaf]
        d = 0
        dims = range(self.ndim+1)
        quit = False
        while (i < self.leaf_stop[curr_leaf]):
            isort = self.leaf_idx_cells[i]
            equal = True
            for d in dims:
                dsort = self.leaf_idx_verts[isort,d]
                if verts[idx_sort[d]] < self.leaf_cells[isort,dsort]:
                    equal = False
                    break
                elif verts[idx_sort[d]] > self.leaf_cells[isort,dsort]:
                    equal = False
                    quit = True
                    break
            if equal:
                return isort
            if quit:
                break
            else:
                i += 1
        # key = tuple(sorted(verts))
        # for curr_cell in xrange(self.leaf_counts[curr_leaf]):
        #     key2 = tuple(sorted(self.get_cell_verts(curr_leaf, curr_cell)))
        #     equal = True
        #     for d in range(self.ndim+1):
        #         if key[d] != key2[d]:
        #             equal = False
        #             break
        #     if equal:
        #         return curr_cell
            # if key == tuple(sorted(self.get_cell_verts(curr_leaf, curr_cell))):
            #     return curr_cell
        return -1

    def append_cell(self, c, n, start=0, stop=None):
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

    def find_leaves(self, c, full=False):
        r"""Determine which leaves originally owned vertices in the set.

        Args:
            c (np.ndarray of int): Vertex indices.
            full (bool, optional): If True, a leaf is recorded for every vertex 
                including `None` for the infinite vertex. Otherwise, a unique 
                set of leaves is returned. Defaults to `False`.

        Returns:
            list of int: Indices of leaves that own one or more of the vertices.

        """
        leaves = []
        for x in c:
            if x == self.idx_inf:
                if full:
                    leaves.append(None)
                continue
            for i in xrange(self.num_leaves):
                leaf = self.tree.leaves[i]
                if x < leaf.stop_idx:
                    leaves.append(i)
                    break
        if full:
            return leaves
        else:
            return list(set(leaves))

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
        self._done = False

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
                time.sleep(0.01)

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
        out = [(leaf.id,leaf.serialize()) for leaf in self._leaves]
        # self._queues[self._proc_idx].put(out)
        self._queues[-1].put(out)
        self._done = True
        
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

    def serialize(self):
        r"""Get the serialized tessellation for this leaf.

        Returns:
            tuple: Vertices and neighbors for cells in the triangulation.

        """
        cells, neigh, idx_inf = self.T.serialize()
        fin = (cells != idx_inf)
        cells[fin] = self.idx[cells[fin]]
        idx_verts, idx_cells = tools.py_arg_sortSerializedTess(cells)
        return cells, neigh, idx_inf, idx_verts, idx_cells



