import cython

import os, pickle, copy
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool as cbool
from cpython cimport bool as pybool
from cython.operator cimport dereference
from cython.operator cimport preincrement
from libc.stdint cimport uint32_t, uint64_t, int32_t, int64_t

# import utils
# from .. import delaunay

# def fname_leaf(leafid):
#     return 'leaf{}.dat'.format(leafid)
# def fname_leaf_tess(leafid):
#     return 'leaf{}_tess.dat'.format(leafid)
# def fname_leaf_out(leafid ,dest):
#     return 'leaf{}_to{}.dat'.format(leafid,dest)

# class Leaf:
#     def __init__(self, int leafid, np.ndarray[np.uint64_t] idx, 
#                  uint64_t npts, uint32_t ndim, 
#                  np.ndarray[np.float64_t] left_edge,
#                  np.ndarray[np.float64_t] right_edge,
#                  np.ndarray[np.float64_t] domain_width,
#                  np.ndarray[np.uint8_t] periodic_left,
#                  np.ndarray[np.uint8_t] periodic_right,
#                  pybool domain_periodic, uint64_t num_leaves):
#         self.idx = idx
#         self.id = leafid
#         self.children = npts
#         self.ndim = ndim
#         self.left_edge = left_edge
#         self.right_edge = right_edge
#         self.domain_width = domain_width
#         self.periodic_left = periodic_left
#         self.periodic_right = periodic_right
#         self.periodic = domain_periodic
#         self.num_leaves = num_leaves
#         self.neighbors = [
#             {'left':[],'right':[],'left_periodic':[],'right_periodic':[]} for i in range(self.ndim)]
                           
#         self.wrapped = np.zeros(self.children, dtype='bool')

#     def tessellate(self):
#         if self.ndim == 2:
#             self.T = Delaunay2()
#         elif self.ndim == 3:
#             self.T = Delaunay3()
#         else:
#             raise Exception('Unsupported number of dimensions: {}'.format(self.ndim))
#         self.T.insert(self.pos)
#         nrep = self.pos.shape[0] - self.num_verts()
#         if nrep != 0:
#             print 'Leaf {} has {} repeated points'.format(self.id,nrep)

#     def num_verts(self):
#         if hasattr(self,'T'):
#             return self.T.num_verts()
#         else:
#             return 0

#     def num_cells(self):
#         if hasattr(self,'T'):
#             return self.T.num_cells()
#         else:
#             return 0

#     def outgoing_points(self):
#         hvall = self.T.outgoing_points(self.left_edge, self.right_edge, self.periodic,
#                                        self.neighbors, self.num_leaves)
#         # Get info for those indices
#         out = [{} for i in range(self.num_leaves)]
#         for i, ind in enumerate(hvall):
#             if len(ind) > 0:
#                 out[i]['idx'] = self.idx[ind]
#                 out[i]['pos'] = self.pos[ind,:]
#                 if self.vel is not None:
#                     out[i]['vel'] = self.vel[ind,:]
#             else:
#                 out[i] = None
#         return out

#     def incoming_points(self, int leafid, object tot):
#         if tot['pos'].shape[0] == 0: return
#         # Wrap points
#         pos = tot['pos']
#         wrapped = np.zeros(pos.shape[0], 'bool')
#         if self.id == leafid:
#             for i in range(self.ndim):
#                 if self.periodic_left[i] and self.periodic_right[i]:
#                     idx_left = (pos[:,i] - self.left_edge[i]) < (self.right_edge[i] - pos[:,i])
#                     idx_right = (self.right_edge[i] - pos[:,i]) < (pos[:,i] - self.left_edge[i])
#                     pos[idx_left,i] += self.domain_width[i]
#                     pos[idx_right,i] -= self.domain_width[i]
#                     wrapped[idx_left] = True
#                     wrapped[idx_right] = True
#         else:
#             for i in range(self.ndim):
#                 if leafid in self.neighbors[i]['right_periodic']:
#                     idx_left = (pos[:,i] + self.domain_width[i] - self.right_edge[i]) < (self.left_edge[i] - pos[:,i]) 
#                     pos[idx_left,i] += self.domain_width[i]
#                     wrapped[idx_left] = True
#                 if leafid in self.neighbors[i]['left_periodic']:
#                     idx_right = (self.left_edge[i] - pos[:,i] + self.domain_width[i]) < (pos[:,i] - self.right_edge[i])
#                     pos[idx_right,i] -= self.domain_width[i]
#                     wrapped[idx_right] = True
#         # Concatenate arrays
#         self.idx = np.concatenate([self.idx, tot['idx']])
#         self.pos = np.concatenate([self.pos, pos], axis=0)
#         if self.vel is not None:
#             if 'vel' not in tot:
#                 raise Exception('Velocities expected, but not recieved from leaf {}'.format(leafid))
#             self.vel = np.concatenate([self.vel, tot['vel']], axis=0)
#         self.wrapped = np.concatenate([self.wrapped, wrapped])
#         # Add points
#         self.T.insert(pos)

#     def voronoi_volumes(self):
#         return self.T.voronoi_volumes(self.children)

#     @property
#     def all_children(self):
#         return self.idx.shape[0]

#     _unpickleable = ['T']

#     def __getstate__(self):
#         for k in self._unpickleable:
#             if k in self.__dict__: del self.__dict__[k]
#         return self.__dict__

#     def outfile(self, dest):
#         return fname_leaf_out(self.id, dest)

#     def infile(self, source):
#         return fname_leaf_out(source, self.id)

#     def to_file(self):
#         self.tess_to_file()
#         fname = fname_leaf(self.id)
#         if os.path.isfile(fname):
#             os.remove(fname)
#         with open(fname, 'wb') as fd:
#             pickle.dump(self, fd)

#     @staticmethod
#     def from_file(leafid):
#         fname = fname_leaf(leafid)
#         with open(fname, 'rb') as fd:
#             out = pickle.load(fd)
#         out.tess_from_file()
#         return out

#     def tess_to_file(self):
#         fname = fname_leaf_tess(self.id)
#         if os.path.isfile(fname):
#             os.remove(fname)
#         if hasattr(self,'T'):
#             self.T.write_to_file(fname)

#     def tess_from_file(self):
#         fname = fname_leaf_tess(self.id)
#         if os.path.isfile(fname):
#             if self.ndim == 2:
#                 T = Delaunay2()
#             elif self.ndim == 3:
#                 T = Delaunay3()
#             else:
#                 raise Exception('Unsupported number of dimensions: {}'.format(self.ndim))
#             T.read_from_file(fname)
#             self.T = T

#     @property
#     def edge_list(self):
#         if not hasattr(self,'_edge_list'):
#             self._edge_list = self.T.edge_info(self.children,self.idx)
#         return self._edge_list

#     @property
#     def edge_dist_pos(self):
#         if not hasattr(self, '_edist_pos'):
#             p1 = self.pos[self.edge_list[:,0],:]
#             p2 = self.pos[self.edge_list[:,1],:]
#             edist = utils.distance(p1, p2)
#             self._edist_pos = edist
#         return self._edist_pos

#     @property
#     def edge_dist_vel(self):
#         if not hasattr(self, '_edist_vel'):
#             if self.vel is None:
#                 raise Exception('Velocity information not provided.')
#             p1 = self.vel[self.edge_list[:,0],:]
#             p2 = self.vel[self.edge_list[:,1],:]
#             edist = utils.distance(p1, p2)
#             self._edist_vel = edist
#         return self._edist_vel

#     def filter_edges(self, method, cut=None):
#         if method == 'pos':
#             if cut is None: cut = np.mean(self.edge_dist_pos)
#             efilter = (self.edge_dist_pos < cut)
#         elif method == 'vel':
#             if cut is None: cut = np.mean(self.edge_dist_vel)
#             efilter = (self.edge_dist_vel > cut)
#         elif method in ('posvel','velpos'):
#             if cut is None:
#                 cut = [np.mean(self.edge_dist_pos), np.mean(self.edge_dist_vel)]
#             else:
#                 assert(isinstance(cut,list))
#                 assert(len(cut) == 2)
#                 if method.startswith('vel'):
#                     cut = [cut[1], cut[0]]
#             efilter = np.logical_and(self.edge_dist_pos < cut[0],
#                                      self.edge_dist_vel > cut[1])
#         else:
#             raise Exception("Invalid filtering method '{}'".format(method))
#         # vfilter = np.zeros(self.all_children,'bool') # Only vertices with edges
#         # vfilter[self.edge_list[efilter,0]] = True
#         # vfilter[self.edge_list[efilter,1]] = True
#         return efilter #, vfilter

#     @property
#     def local_scaling_pos(self):
#         earr = self.edge_list
#         edist = self.edge_dist_pos
#         msig = np.zeros(self.all_children, 'float')
#         for i in xrange(earr.shape[0]):
#             v1 = earr[i,0]
#             v2 = earr[i,1]
#             msig[v1] = max(msig[v1],edist[i])
#             msig[v2] = max(msig[v2],edist[i])
#         return msig

#     @property
#     def local_scaling_vel(self):
#         earr = self.edge_list
#         edist = self.edge_dist_vel
#         msig = 9999*np.ones(self.all_children, 'float')
#         for i in xrange(earr.shape[0]):
#             v1 = earr[i,0]
#             v2 = earr[i,1]
#             msig[v1] = min(msig[v1],edist[i])
#             msig[v2] = min(msig[v2],edist[i])
#         return msig

#     @property
#     def edge_arr(self):
#         edge_list = self.edge_list
#         edge_arr = np.zeros(edge_list.shape, dtype='int64')
#         edge_arr[:,0] = self.idx[edge_list[:,0]]
#         edge_arr[:,1] = self.idx[edge_list[:,1]]
#         return edge_arr

#     @property
#     def wrapped_edges(self):
#         if not hasattr(self,'_wrapped_edge_arr'):
#             wrapped = {}
#             pts = []
#             old_idx = []
#             npts = 0
#             edge_list = self.edge_list
#             edge_arr = np.zeros(edge_list.shape, dtype='int64')
#             edge_arr[:,0] = self.idx[edge_list[:,0]]
#             edge_arr[:,1] = self.idx[edge_list[:,1]]

#             hvwrap = utils.hash_array(self.pos[edge_list[:,1],:])
#             idx = self.wrapped[edge_list[:,1]]
#             for i in np.where(idx)[0]:
#                 hv2 = hvwrap[i]
#                 if not hv2 in wrapped:
#                     wrapped[hv2] = npts
#                     pts.append(self.pos[edge_list[i,1],:])
#                     old_idx.append(self.idx[edge_list[i,1]])
#                     npts += 1
#                 edge_arr[i,1] = wrapped[hv2]
#             self._wrapped_edge_arr = edge_arr
#             if npts > 0:
#                 self._wrapped_points = np.vstack(pts)
#             else:
#                 self._wrapped_points = np.zeros((0,self.ndim), dtype='float64')
#             self._wrapped_old_idx = np.array(old_idx)
#             self._wrapped_index = idx
#         return self._wrapped_edge_arr, self._wrapped_points, self._wrapped_old_idx, self._wrapped_index

# def test_pivot(int N, int ndim, int d):
#     np.random.seed(10)
#     cdef np.ndarray[np.float64_t, ndim=2] x = np.random.rand(N,ndim).astype('float64')
#     print 'before: {}'.format(x)
#     cdef np.ndarray[np.uint64_t, ndim=1] idx = np.arange(N).astype('uint64')
#     cdef np.int64_t l = 0
#     cdef np.int64_t r = N-1
#     cdef np.int64_t p = (N-1)/2
#     p = pivot(&x[0,0], &idx[0], ndim, d, l, r)
#     print 'after: {}'.format(x[idx,:])
#     print 'x[{},{}] = {}'.format(p,d,x[idx[p],d])
#     # assert((x[idx[:(p+1)],d] <= med).all())
#     # assert((x[idx[(p+1):],d] > med).all())
#     # assert(x[idx[p],d] == med)

# def median(np.ndarray[np.float64_t, ndim=2] pos, np.uint32_t d):
#     cdef np.uint64_t npts
#     cdef np.uint32_t ndim
#     npts, ndim = pos.shape[0], pos.shape[1]
#     cdef np.ndarray[np.uint64_t, ndim = 1] idx = np.arange(npts).astype('uint64')
#     cdef np.int64_t l = 0
#     cdef np.int64_t r = npts-1
#     cdef np.int64_t p = (r+l)/2
#     p = select(&pos[0,0], &idx[0], ndim, d, l, r, p)
#     cdef np.float64_t med
#     if (npts%2) == 0:
#         med = pos[idx[p],d]
#         # med = (pos[idx[p],d] + pos[idx[p+1],d])/2.0
#     else:
#         med = pos[idx[p],d]
#     return med

# cdef class PyKDTree:
#     def __cinit__(self, np.ndarray[double, ndim=2] pts, 
#                   np.ndarray[double, ndim=1] left_edge,
#                   np.ndarray[double, ndim=1] right_edge,
#                   pybool periodic,
#                   np.uint32_t leafsize = 10000,
#                   np.ndarray[double, ndim=2] vel = None):
#         cdef int n, m, k
#         n, m = pts.shape[0], pts.shape[1]
#         self.npts = <uint64_t>n
#         self.ndim = <uint32_t>m
#         self.leafsize = <uint32_t>leafsize
#         self.left_edge = &left_edge[0]
#         self.right_edge = &right_edge[0]
#         cdef np.ndarray[double] domain_width = right_edge - left_edge
#         self.domain_width = &domain_width[0]
#         self.periodic = <cbool>periodic
#         cdef np.ndarray[np.uint64_t] idx = np.arange(self.npts).astype('uint64')
#         self.tree = new KDTree(&pts[0,0], &idx[0], self.npts, self.ndim,
#                                self.leafsize, self.left_edge, self.right_edge)
#         self.leaves = []
#         self.num_leaves = self.tree.leaves.size()
#         cdef object leaf
#         for k in xrange(self.num_leaves):
#             leaf = self.process_leaf(k,idx)
#             leaf.pos = copy.copy(pts[leaf.idx,:])
#             if vel is not None:
#                 leaf.vel = copy.copy(vel[leaf.idx,:])
#             else:
#                 leaf.vel = None

#     def process_leaf(self, int leafid, np.ndarray[np.uint64_t] all_idx):
#         cdef Node* leafnode = self.tree.leaves[leafid]
#         cdef np.ndarray[np.uint8_t] periodic_left = np.zeros(self.ndim, dtype='uint8')
#         cdef np.ndarray[np.uint8_t] periodic_right = np.zeros(self.ndim, dtype='uint8')
#         cdef int i
#         if self.periodic:
#             for i in range(<int>self.ndim):
#                 if np.isclose(leafnode.left_edge[i], self.left_edge[i]):
#                     periodic_left[i] = 1
#                 if np.isclose(leafnode.right_edge[i], self.right_edge[i]):
#                     periodic_right[i] = 1
#         # Indices
#         cdef np.ndarray[np.uint64_t] idx
#         idx = all_idx[leafnode.left_idx:(leafnode.left_idx + leafnode.children)]
#         # cdef np.npy_intp shape[1]
#         # shape[0] = <np.npy_intp>leafnode.children
#         # idx = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT64, leafnode.idx)
#         # Edges
#         cdef np.ndarray[np.float64_t] left_edge = np.zeros(self.ndim, 'float64')
#         cdef np.ndarray[np.float64_t] right_edge = np.zeros(self.ndim, 'float64')
#         cdef np.ndarray[np.float64_t] domain_width = np.zeros(self.ndim, 'float64')
#         for i in range(<int>self.ndim):
#             left_edge[i] = leafnode.left_edge[i]
#             right_edge[i] = leafnode.right_edge[i]
#             domain_width[i] = self.domain_width[i]
#         # Create leaf
#         assert(len(idx) == leafnode.children)
#         cdef object leaf = Leaf(leafid, idx, leafnode.children, self.ndim,
#                                 left_edge, right_edge, domain_width,
#                                 periodic_left, periodic_right, self.periodic, 
#                                 self.num_leaves)
#         self.leaves.append(leaf)
#         # Find neighbors in previous leaves
#         cdef Node* prevnode
#         cdef object matches
#         cdef pybool match
#         # print 'Leaf {}'.format(leaf.id)
#         # print '  LE: {}, {}'.format(leaf.left_edge[0], leaf.left_edge[1])
#         # print '  RE: {}, {}'.format(leaf.right_edge[0], leaf.right_edge[1])
#         for prev in self.leaves:
#             prevnode = self.tree.leaves[prev.id]
#             matches = self.ndim*[None]
#             match = True
#             # Check that there is overlap in all dimensions
#             for i in range(<int>self.ndim):
#                 if leaf.left_edge[i] > prev.right_edge[i]:
#                 # if leafnode.left_edge[i] > prevnode.right_edge[i]:
#                     if self.periodic and leaf.periodic_right[i] and prev.periodic_left[i]:
#                         pass
#                     else:
#                         match = False
#                         break
#                 if leaf.right_edge[i] < prev.left_edge[i]:
#                 # if leafnode.right_edge[i] < prevnode.left_edge[i]:
#                     if self.periodic and prev.periodic_right[i] and leaf.periodic_left[i]:
#                         pass
#                     else:
#                         match = False
#                         break
#             # Add neighbors sharing boundaries
#             # if prev.id == 0:
#             #     print '    Prev 0, match = {}'.format(match)
#             #     print '      LE: {}, {}'.format(prev.left_edge[0], prev.left_edge[1])
#             #     print '      RE: {}, {}'.format(prev.right_edge[0], prev.right_edge[1])
#             if match:
#                 for i in range(<int>self.ndim):
#                     if np.isclose(leafnode.left_edge[i], prevnode.right_edge[i]):
#                         leaf.neighbors[i]['left'].append(prev.id)
#                         prev.neighbors[i]['right'].append(leaf.id)
#                     elif np.isclose(leafnode.right_edge[i], prevnode.left_edge[i]):
#                         leaf.neighbors[i]['right'].append(prev.id)
#                         prev.neighbors[i]['left'].append(leaf.id)
#                     if self.periodic and leaf.periodic_right[i] and prev.periodic_left[i]:
#                         leaf.neighbors[i]['right_periodic'].append(prev.id)
#                         prev.neighbors[i]['left_periodic'].append(leaf.id)
#                     if self.periodic and prev.periodic_right[i] and leaf.periodic_left[i]:
#                         leaf.neighbors[i]['left_periodic'].append(prev.id)
#                         prev.neighbors[i]['right_periodic'].append(leaf.id)
#         return leaf
