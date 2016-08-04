import numpy as np

from cgal4py import domain_decomp

N = 100 ; leafsize = 10
pts2 = np.random.rand(N,2).astype('float64')
left_edge2 = np.zeros(2, 'float64')
right_edge2 = np.ones(2, 'float64')
pts3 = np.random.rand(N,3).astype('float64')
left_edge3 = np.zeros(3, 'float64')
right_edge3 = np.ones(3, 'float64')

def test_Leaf():
    leaf2 = domain_decomp.Leaf(0, np.arange(N), left_edge2, right_edge2)
    leaf3 = domain_decomp.Leaf(0, np.arange(N), left_edge3, right_edge3)

def test_kdtree():
    leaves2 = domain_decomp.kdtree(pts2, left_edge2, right_edge2, leafsize)
    leaves3 = domain_decomp.kdtree(pts3, left_edge3, right_edge3, leafsize)

def test_leaves():
    leaves2 = domain_decomp.leaves('kdtree',pts2, left_edge2, right_edge2, leafsize)
    leaves3 = domain_decomp.leaves('kdtree',pts3, left_edge3, right_edge3, leafsize)
