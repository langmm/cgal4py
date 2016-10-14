import numpy as np
import time, copy, os
import cProfile, pstats
np.random.seed(10)
from cgal4py import parallel, delaunay
from test_cgal4py import make_points, make_test
import matplotlib.pyplot as plt

def run(func,npart, nproc, ndim):
    if nproc == 1:
        pts, le, re = make_points(npart, ndim)
        if func == 'Delaunay':
            T = delaunay.Delaunay(pts)
        elif func == 'VoronoiVolumes':
            vol = delaunay.VoronoiVolumes(pts)
        else:
            raise ValueError("Unsupported function: {}".format(func))
    else:
        pts, tree = make_test(npart, ndim, nleaves=nproc)
        if func == 'Delaunay':
            T = parallel.ParallelDelaunay(pts, tree, nproc)
        elif func == 'VoronoiVolumes':
            vol = parallel.ParallelVoronoiVolumes(pts, tree, nproc)
        else:
            raise ValueError("Unsupported function: {}".format(func))

def stats_run(func, npart, nproc, ndim, overwrite=False):
    fname_stat = 'stat_{}_{}part_{}proc_{}dim.txt'.format(func, npart, nproc, ndim)
    if overwrite or not os.path.isfile(fname_stat):
        cProfile.run('run({},{},{},{})'.format(func,npart,nproc,ndim), fname_stat)
    return fname_stat

def time_run(func, npart, nproc, ndim, nrep=1):
    times = np.empty(nrep,'float')
    for i in range(nrep):
        t1 = time.time()
        run(func, npart, nproc, ndim)
        t2 = time.time()
        times[i] = t2 - t1
    return np.mean(times), np.std(times)

def scaling_nproc(func, npart=1e6, nrep=1, overwrite=True):
    npart = int(npart)
    fname_plot = 'plot_scaling_{}_nproc_{}part.png'.format(func, npart)
    nproc_list = [1, 2, 4, 8, 16]
    ndim_list = [2,3]
    clr_list = ['b','r']
    times = np.empty((len(nproc_list),len(ndim_list),2),'float')
    for j,nproc in enumerate(nproc_list):
        for i,ndim in enumerate(ndim_list):
            times[j,i,0],times[j,i,1] = time_run(func, npart, nproc, ndim, nrep=nrep)
    fig, axs = plt.subplots(1,1)
    for i in range(len(ndim_list)):
        ndim = ndim_list[i]
        clr = clr_list[i]
        axs.errorbar(nproc_list, times[:,i,0], yerr=times[:,i,1],
                     fmt=clr, label='ndim = {}'.format(ndim))
    axs.set_xlabel("# of Processors")
    axs.set_ylabel("Time (s)")
    axs.legend()
    fig.savefig(fname_plot)
    print '    '+fname_plot

