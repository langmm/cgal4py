############
Installation
############

Installing Dependencies
=======================

|cgal4py| requires the `CGAL`_ |cgalver| c++ library and the `cykdtree`_

Installing `CGAL`_
------------------

Directions for installing `CGAL`_ |cgalver| can be found `here <http://doc.cgal.org/latest/Manual/installation.html>`_. It should be noted that, as of the latest |cgal4py| release, only packages of `CGAL`_ |cgalver| for Debian unstable (sid) are available from the official Debian repository (See `CGAL FAQ <http://www.cgal.org/FAQ.html#debian_packages>`_). If the version in the repository you are using is not |cgalver|, please `download <http://doc.cgal.org/latest/Manual/installation.html#title6>`_ and `build <http://doc.cgal.org/latest/Manual/installation.html#title0>`_ the `CGAL`_ libraries manually. This requires that you have `cmake <https://cmake.org/>`_ and `checkinstall <http://asic-linux.com.mx/~izto/checkinstall/>`_ installed, both of which can be ``apt-get`` installed.::

   $ cd CGAL-4.9 # go to CGAL directory
   $ cmake . # configure CGAL
   $ make # build the CGAL libraries
   $ checkinstall # install the CGAL libraries

Use of older versions of `CGAL`_ will disable some features of |cgal4py| (e.g. periodic boundary conditions on a single processor).
   
Installing `cykdtree`_
----------------------

..todo:: Links to PyPI release

`cykdtree`_ can be installed from either PyPI using ``pip``::

   $ pip install cykdtree

or by cloning the `Mercurial <https://mercurial.selenic.com/>`_ Bitbucket `repository <https://bitbucket.org/langmm/cykdtree>`_::

   $ hg clone https://[username]@bitbucket.org/[username]/cykdtree

and then building the distribution.::

   $ cd cykdtree
   $ python setup.py install

If you do not have admin privileges on the target machine, ``--user`` can be added to either the ``pip`` or ``setup.py`` installation commands.

Installing |cgal4py|
====================

..todo:: Links to PyPI release

|cgal4py| can be installed from either PyPI using ``pip``::

   $ pip install cgal4py

or by cloning the `Mercurial <https://mercurial.selenic.com/>`_ Bitbucket `repository <https://bitbucket.org/langmm/cgal4py>`_::

   $ hg clone https://[username]@bitbucket.org/[username]/cgal4py

and then building the distribution (it may take a while to compile all of the Cython extensions).::

   $ cd cgal4py
   $ python setup.py install

If you do not have admin privileges on the target machine, ``--user`` can be added to either the ``pip`` or ``setup.py`` installation commands.

Testing the Installation
========================

To test that everything was installed propertly. From the python prompt, import |cgal4py|::

   >>> import cgal4py

and try to access the documentation::

   >>> help(cgal4py)

Additional tests are available and can be run from the command line using `nose <http://nose.readthedocs.io/en/latest/>`_::

   $ cd cgal4py
   $ nosetests

