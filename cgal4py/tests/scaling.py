r"""Routines for tracking the scaling of the triangulation routines."""
import numpy as np
import time
import os
import cProfile
# import pstats
from cgal4py import parallel, delaunay
from test_cgal4py import make_points, make_test
import matplotlib.pyplot as plt
np.random.seed(10)


def run(func, npart, nproc, ndim):
    r"""Run a rountine with a designated number of points & dimensions on a
    selected number of processors.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int): Number of particles.
        nproc (int): Number of processors.
        ndim (int): Number of dimensions.

    Raises:
        ValueError: If `func` is not one of the supported values.

    """
    if nproc == 1:
        pts, le, re = make_points(npart, ndim)
        if func == 'Delaunay':
            delaunay.Delaunay(pts)
        elif func == 'VoronoiVolumes':
            delaunay.VoronoiVolumes(pts)
        else:
            raise ValueError("Unsupported function: {}".format(func))
    else:
        pts, tree = make_test(npart, ndim, nleaves=nproc)
        if func == 'Delaunay':
            parallel.ParallelDelaunay(pts, tree, nproc)
        elif func == 'VoronoiVolumes':
            parallel.ParallelVoronoiVolumes(pts, tree, nproc)
        else:
            raise ValueError("Unsupported function: {}".format(func))


def stats_run(func, npart, nproc, ndim, overwrite=False):
    r"""Get timing stats using :package:`cProfile`.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int): Number of particles.
        nproc (int): Number of processors.
        ndim (int): Number of dimensions.
        overwrite (bool, optional): If True, the existing file for this
            set of input parameters if overwritten. Defaults to False.

    """
    fname_stat = 'stat_{}_{}part_{}proc_{}dim.txt'.format(func, npart,
                                                          nproc, ndim)
    if overwrite or not os.path.isfile(fname_stat):
        cProfile.run('run({},{},{},{})'.format(func, npart, nproc, ndim),
                     fname_stat)
    return fname_stat


def time_run(func, npart, nproc, ndim, nrep=1):
    r"""Get runing times using :package:`time`.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int): Number of particles.
        nproc (int): Number of processors.
        ndim (int): Number of dimensions.
        nrep (int, optional): Number of times the run should be performed to
            get an average. Defaults to 1.

    """
    times = np.empty(nrep, 'float')
    for i in range(nrep):
        t1 = time.time()
        run(func, npart, nproc, ndim)
        t2 = time.time()
        times[i] = t2 - t1
    return np.mean(times), np.std(times)


def scaling_nproc(func, npart=1e6, nrep=1, overwrite=True):
    r"""Plot the scaling with number of processors for a particular function.

    Args:
        func (str): Name of the function that should be run. Values include:
            'Delaunay': Full triangulation.
            'VoronoiVolumes': Cell volumes from triangulation.
        npart (int, optional): Number of particles. Defaults to 1e6.
        nrep (int, optional): Number of times the run should be performed to
            get an average. Defaults to 1.
        overwrite (bool, optional): If True, the existing file for this
            set of input parameters if overwritten. Defaults to False.

    """
    npart = int(npart)
    fname_plot = 'plot_scaling_{}_nproc_{}part.png'.format(func, npart)
    nproc_list = [1, 2, 4, 8, 16]
    ndim_list = [2, 3]
    clr_list = ['b', 'r']
    times = np.empty((len(nproc_list), len(ndim_list), 2), 'float')
    for j, nproc in enumerate(nproc_list):
        for i, ndim in enumerate(ndim_list):
            times[j, i, 0], times[j, i, 1] = time_run(func, npart, nproc, ndim,
                                                      nrep=nrep)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(ndim_list)):
        ndim = ndim_list[i]
        clr = clr_list[i]
        axs.errorbar(nproc_list, times[:, i, 0], yerr=times[:, i, 1],
                     fmt=clr, label='ndim = {}'.format(ndim))
    axs.set_xlabel("# of Processors")
    axs.set_ylabel("Time (s)")
    axs.legend()
    fig.savefig(fname_plot)
    print('    '+fname_plot)
