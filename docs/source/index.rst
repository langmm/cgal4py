.. cgal4py documentation master file, created by
   sphinx-quickstart on Wed Jul 27 21:10:14 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cgal4py's documentation!
===================================

Contents:

.. toctree::
   :maxdepth: 2

   code

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. todo::
   * class for each process that contains leaf/leaves and handles communication
   * general class for domain decomp that returns full list of leaves
   * for mpi parallelism, each node gets base class
   * for thread parallelism, single base class, separate class for each process
   * General structure:
      * MPI parallelism:
	 1. Do domain decomp on process 0
	 2. Communicate leaves to each process (one class per process)
	 3. Perform triangulation on each process
	 4. Do communication of points at edges
	 5. Provide interface to total process
      * Thread prallelism:
	 1. Do domain decomp
	 2. Split leaves between threads
	 3. Perform triangulation on each thread
	 4. Do communication between triangulations/combine triangulations
	 5. Provide interface to total triangulation
