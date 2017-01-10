======
README
======

cgal4py is a Python interface for using the `CGAL <http://www.cgal.org>`__ Delaunay triangulation classes in any number of dimensions. Triangulation in parallel is also supported using the algorithm described in `Peterka, Morozov, & Phillips (2014) <http://mrzv.org/publications/distributed-delaunay/>`_. Documentation for cgal4py can be found `here <http://cgal4py.readthedocs.io/en/latest/>`_.

---------
Licensing
---------
cgal4py is released as open source software under a BSD license.

------------
Requirements
------------
For running in serial:

 * `Python 2.7 <https://www.python.org/download/releases/2.7/>`_
 * C++14 compiler
 * `Cython <http://cython.org/>`_
 * `CGAL <http://www.cgal.org/download.html>`__ Version 3.5 or higher is required for periodic triangulations in 3D and version 4.9 or higher is required for periodic triangulations in 2D.

For running in parallel you will need the above plus:

 * MPI (either `MPICH <https://www.mpich.org/>`_ or `OpenMPI <https://www.open-mpi.org/>`_)
 * `mpi4py <http://pythonhosted.org/mpi4py/>`_
 * `multiprocessing <https://docs.python.org/2/library/multiprocessing.html>`_
 * `cykdtree <https://bitbucket.org/langmm/cykdtree>`_

------------
Installation
------------

From Source
===========
1. Clone the cgal4py package using `Mercurial <https://www.mercurial-scm.org/>`_. ``$ hg clone https://[username]@bitbucket.org/[username]/cgal4py`` 
where ``[username]`` should be replaced with your Bitbucket username. 
2. From the distribution directory, execute the install script. ``$ python setup.py install`` If you do not have administrative privileges, add the flag ``--user`` to the above command and the package will be installed in your `user package directory <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.

-----------------
Who do I talk to?
-----------------
This package is currently maintained by `Meagan Lang <mailto:langmm.astro@gmail.com>`_.
