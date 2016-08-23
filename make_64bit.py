
def main():
    fname32 = 'cgal4py/delaunay/delaunay2.pyx'
    fname64 = 'cgal4py/delaunay/delaunay2_64bit.pyx'

    replace = [["ctypedef uint32_t info_t","ctypedef uint64_t info_t"],
               ["cdef object np_info = np.uint32","cdef object np_info = np.uint64"],
               ["ctypedef np.uint32_t np_info_t","ctypedef np.uint64_t np_info_t"]]

    with open(fname64,'w') as new_file:
        with open(fname32,'r') as old_file:
            for line in old_file:
                if replace[0][0] in line:
                    new_file.write(line.replace(replace[0][0],replace[0][1]))
                elif replace [1][0] in line:
                    new_file.write(line.replace(replace[1][0],replace[1][1]))
                else:
                    new_file.write(line)
                
if __name__ == "__main__":
    main()
