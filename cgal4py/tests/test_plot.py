from nose import with_setup
import numpy as np
import os
from cgal4py.delaunay import Delaunay2, Delaunay3
from test_delaunay2 import pts as pts2
from test_delaunay3 import pts as pts3

def test_plot2D():
    fname_test = "test_plot2D.png"
    T = Delaunay2()
    T.insert(pts2)
    axs = T.plot(plotfile=fname_test, title='Test')
    os.remove(fname_test)
    # T.plot(axs=axs)

def test_plot3D():
    fname_test = "test_plot3D.png"
    T = Delaunay3()
    T.insert(pts3)
    axs = T.plot(plotfile=fname_test, title='Test')
    os.remove(fname_test)
    # T.plot(axs=axs)
