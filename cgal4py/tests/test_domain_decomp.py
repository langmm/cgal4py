r"""Tests for domain decomposition methods."""
import numpy as np
from nose.tools import assert_raises
from cgal4py import domain_decomp


N = 100
leafsize = 10
pts2 = np.random.rand(N, 2).astype('float64')
left_edge2 = np.zeros(2, 'float64')
right_edge2 = np.ones(2, 'float64')
pts3 = np.random.rand(N, 3).astype('float64')
left_edge3 = np.zeros(3, 'float64')
right_edge3 = np.ones(3, 'float64')


def test_tree():
    tree2 = domain_decomp.tree('kdtree', pts2, left_edge2, right_edge2,
                               periodic=False, leafsize=leafsize)
    tree3 = domain_decomp.tree('kdtree', pts3, left_edge3, right_edge3,
                               periodic=False, leafsize=leafsize)
    assert_raises(ValueError, domain_decomp.tree, 'invalid',
                  pts2, left_edge2, right_edge2, False)
    del tree2, tree3


def test_GenericLeaf():
    leaf2 = domain_decomp.GenericLeaf(N, left_edge2, right_edge2)
    leaf3 = domain_decomp.GenericLeaf(N, left_edge3, right_edge3)
    del leaf2, leaf3


def test_process_leaves():
    # With information
    tree2 = domain_decomp.tree('kdtree', pts2, left_edge2, right_edge2,
                               periodic=False, leafsize=leafsize)
    tree3 = domain_decomp.tree('kdtree', pts3, left_edge3, right_edge3,
                               periodic=False, leafsize=leafsize)
    leaves2 = domain_decomp.process_leaves(tree2.leaves, left_edge2,
                                           right_edge2, False)
    leaves3 = domain_decomp.process_leaves(tree3.leaves, left_edge3,
                                           right_edge3, False)
    # Without information
    left_edges = np.array([[0.0, 0.0],
                           [0.0, 0.5],
                           [0.5, 0.0],
                           [0.5, 0.5]], 'float')
    right_edges = np.array([[0.5, 0.5],
                            [0.5, 1.0],
                            [1.0, 0.5],
                            [1.0, 1.0]], 'float')
    leaves = [domain_decomp.GenericLeaf(
        5, left_edges[i, :], right_edges[i, :]) for i in range(2)]
    leaves = domain_decomp.process_leaves(leaves, left_edge2,
                                          right_edge2, False)
    del leaves2, leaves3


def test_GenericTree():
    # With information
    tree2 = domain_decomp.tree('kdtree', pts2, left_edge2, right_edge2,
                               periodic=False, leafsize=leafsize)
    tree3 = domain_decomp.tree('kdtree', pts3, left_edge3, right_edge3,
                               periodic=False, leafsize=leafsize)
    leaves2 = domain_decomp.process_leaves(tree2.leaves, left_edge2,
                                           right_edge2, False)
    leaves3 = domain_decomp.process_leaves(tree3.leaves, left_edge3,
                                           right_edge3, False)
    # Without information
    left_edges = np.array([[0.0, 0.0],
                           [0.0, 0.5],
                           [0.5, 0.0],
                           [0.5, 0.5]], 'float')
    right_edges = np.array([[0.5, 0.5],
                            [0.5, 1.0],
                            [1.0, 0.5],
                            [1.0, 1.0]], 'float')
    leaves = [domain_decomp.GenericLeaf(
        5, left_edges[i, :], right_edges[i, :]) for i in range(2)]
    tree = domain_decomp.GenericTree(
        np.arange(5*left_edges.shape[0]).astype('int'),
        leaves, left_edge2, right_edge2, False)
    del tree, leaves2, leaves3
