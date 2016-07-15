from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += cythonize(Extension("cgal4py/delaunay2",
                             sources=["cgal4py/delaunay2.pyx","cgal4py/c_delaunay2.cpp"],
                             language="c++",
                             include_dirs=[numpy.get_include()],
                             libraries=['gmp'],
                             extra_link_args=["-lgmp"],
                             extra_compile_args=["-std=gnu++11"]))
    ext_modules += cythonize(Extension("cgal4py/delaunay3",
                             sources=["cgal4py/delaunay3.pyx","cgal4py/c_delaunay3.cpp"],
                             language="c++",
                             include_dirs=[numpy.get_include()],
                             libraries=['gmp'],
                             extra_link_args=["-lgmp"],
                             extra_compile_args=["-std=gnu++11"]))
    # ext_modules += cythonize(Extension("cgal4py/kdtree",
    #                          sources=["cgal4py/kdtree.pyx","cgal4py/c_kdtree.cpp"],
    #                          language="c++",
    #                          include_dirs=[numpy.get_include()],
    #                          extra_compile_args=["-std=gnu++11"]))
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("cgal4py.delaunay2", ["cgal4py/c_delaunay2.cpp"],
                  include_dirs=[numpy.get_include()]),
        Extension("cgal4py.delaunay3", ["cgal4py/c_delaunay3.cpp"],
                  include_dirs=[numpy.get_include()]),
    ]

setup(name='cgal4py',
      version='0.1',
      description='Python interface for CGAL Triangulations',
      url='https://langmm@bitbucket.org/langmm/cgal4py',
      author='Meagan Lang',
      author_email='langmm.astro@gmail.com',
      license='GPL',
      packages=['cgal4py'],
      zip_safe=False,
      cmdclass = cmdclass,
      ext_modules = ext_modules)




# setup(
#     ext_modules=[
#         Extension("mlutils", ["mlutils.c"],
#                   include_dirs=[])#umpy.get_include()]),
#     ],
# )

# # from Cython.Build import cythonize
# setup(
#     ext_modules = cythonize("mlutils.pyx")
# )
