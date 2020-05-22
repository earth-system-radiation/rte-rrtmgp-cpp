# distutils: language = c++
import cython
from libcpp.string cimport string as std_string
import numpy as np
cimport numpy as np

cdef extern from *:
    ctypedef int d1 "1"
    ctypedef int d2 "2"
    ctypedef int d3 "3"

cdef extern from "<array>" namespace "std":
    cdef cppclass array[T, ND]:
        array() except+
        int& operator[](size_t)

cdef extern from "../include/Array.h":
    cdef cppclass Array[T, ND]:
        Array() except+
        void set_dims(const array[int, ND]&) except +
        T& operator()(const array[int, ND]&)

cdef extern from "../include_test/Radiation_solver.h":
    cdef cppclass Radiation_solver[TF]:
        Radiation_solver() except +
        void load_kdistribution_lw(const std_string&);
        void set_vmr(const std_string&, const TF);
        void set_vmr(const std_string&, const Array[TF,d1]&);
        void set_vmr(const std_string&, const Array[TF,d2]&);

def get_array():
    cdef array[int, d2] dims
    dims[0], dims[1] = 3, 5
    cdef Array[double, d2] a
    a.set_dims(dims)

    cdef array[int, d2] index
    cdef double* d

    counter = 0
    for j in range(dims[1]):
        for i in range(dims[0]):
            index[0], index[1] = i+1, j+1
            d = &a(index)
            d[0] = counter
            counter += 1.

    a_np = np.zeros((dims[1], dims[0]))

    for j in range(dims[1]):
        for i in range(dims[0]):
            index[0], index[1] = i+1, j+1
            a_np[j,i] = a(index)

    return a_np

