import numpy as np
import time, copy, os
import cProfile, pstats
np.random.seed(10)
from cgal4py import parallel, delaunay
from test_cgal4py import make_test
import matplotlib.pyplot as plt

def run(npart, nproc, ndim):
    if nproc == 1:
        T = delaunay.Delaunay(pts)
    else:
        pts, tree = make_test(npart, ndim, nleaves=nproc)
        T = parallel.ParallelDelaunay(pts, tree, nproc)

def stats_run(npart, nproc, ndim, overwrite=False):
    fname_stat = 'stat_{}part_{}proc_{}dim.txt'.format(npart, nproc, ndim)
    if overwrite or not os.path.isfile(fname_stat):
        cProfile.run('run({},{},{})', fname_stat)
    return fname_stat

def time_run(npart, nproc, ndim):
    t1 = time.time()
    run(npart, nproc, ndim)
    t2 = time.time()
    return t2-t1

def scaling_nproc(npart=1e6, overwrite=True):
    npart = int(npart)
    fname_plot = 'plot_scaling_nproc_{}part.png'.format(npart)
    nproc_list = [1, 2, 4, 8, 16]
    ndim_list = [2]#,3]
    times_lists = []
    for ndim in ndim_list:
        times_lists.append([time_run(npart, nproc, ndim) for nproc in nproc_list])
    fig, axs = plt.subplots(1,1)
    for ndim, times_list in zip(ndim_list, times_lists):
        axs.plot(nproc_list, times_list, label='ndim = {}'.format(ndim))

