# README #

cgal4py is a Python interface for using the [CGAL](http://www.cgal.org) Delaunay triangulation classes in any number of dimensions. Triangulation in parallel is also supported using the algorithm described in [Peterka, Morozov, & Phillips (2014)](http://mrzv.org/publications/distributed-delaunay/). Documentation for cgal4py can be found [here](http://cgal4py.readthedocs.io/en/latest/).

## Licensing ##
cgal4py is released as open source software under a BSD license.

## Requirements ##
For running in serial:

* [Python 2.7](https://www.python.org/download/releases/2.7/)
* C++14 compiler
* [Cython](http://cython.org/)
* [CGAL](http://www.cgal.org/download.html) Version 3.5 or higher is required for periodic triangulations in 3D and version 4.9 or higher is required for periodic triangulations in 2D.

For running in parallel you will need the above plus:

* MPI (either [MPICH](https://www.mpich.org/) or [OpenMPI](https://www.open-mpi.org/))
* [mpi4py](http://pythonhosted.org/mpi4py/)
* [multiprocessing](https://docs.python.org/2/library/multiprocessing.html)
* [cykdtree](https://github.com/cykdtree/cykdtree)

## Installation ##

### From Source ###
1. Clone the cgal4py package using [git](https://git-scm.com/).
```$ git clone https://github.com/langmm/cgal4py.git```
2. From the distribution directory, execute the install script. ```$ python setup.py install```
If you do not have administrative privileges, add the flag ```--user``` to the above command and the package will be installed in your [user package directory](https://docs.python.org/2/install/#alternate-installation-the-user-scheme).

## Who do I talk to? ##
This package is currently maintained by [Meagan Lang](mailto:langmm.astro@gmail.com)
