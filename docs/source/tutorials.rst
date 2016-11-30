#########
Tutorials
#########

Delaunay Triangulation
======================

Getting a Delaunay triangulation for a 2D or 3D point distribution using |cgal4py| is straight forward.::

   >>> import cgal4py
   >>> import numpy as np
   >>> pts_2D = np.random.random(100,2)
   >>> T_2D = cgal4py.Delaunay(pts_2D)
   >>> pts_3D = np.random.random(100,3)
   >>> T_3D = cgal4py.Delaunay(pts_3D)

Periodic Delaunay Triangulation
-------------------------------

To get a triangulation for a distribution with periodic boundary conditions, you must also provide the boundary conditions::

   >>> pts = np.random.random(100,2)
   >>> LE = np.zeros(2, 'float')
   >>> RE = np.ones(2, 'float')
   >>> T_Per = cgal4py.Delaunay(pts, periodic=True, left_edge=LE, right_edge=RE)

Parallel Delaunay Triangulation
-------------------------------

Parallel triangulation is just as easy.::

   >>> pts = np.random.random(100,2) 
   >>> T_Para = cgal4py.Delaunay(pts, nproc=2)

Traversing a Triangulation
==========================

Iterating Over Vertices
-----------------------

Iterating Over Cells
--------------------

Iterating Over Edges
--------------------

Voronoi Volumes
===============


