# distutils: language = c++
import cython
from libcpp.string cimport string as std_string
import numpy as np
cimport numpy as np
import numbers


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


# ctypedef fused float_t:
#     cython.float
#     cython.double


cdef class Radiation_solver_wrapper:
    cdef Radiation_solver[double] rad

    def set_vmr(self, gas_name, gas_conc):
        cdef std_string gas_name_cpp = gas_name

        cdef double gas_conc_0d

        cdef Array[double, d1] gas_conc_1d
        cdef Array[double, d2] gas_conc_2d

        cdef array[int, d1] dims_1d
        cdef array[int, d1] index_1d

        cdef array[int, d2] dims_2d
        cdef array[int, d2] index_2d

        cdef double* gas_conc_ptr

        if isinstance(gas_conc, np.ndarray):

            if len(gas_conc.shape) == 0:
                gas_conc_0d = gas_conc
                self.rad.set_vmr(gas_name_cpp, gas_conc_0d)

            elif len(gas_conc.shape) == 1:
                dims_1d[0] = gas_conc.shape[0]
                gas_conc_1d.set_dims(dims_1d)

                # Adjust for the Fortran indexing used in Array.
                for ilev in range(gas_conc.shape[0]):
                    index_1d[0] = ilev+1
                    gas_conc_ptr = &gas_conc_1d(index_1d)
                    gas_conc_ptr[0] = gas_conc[ilev]

                self.rad.set_vmr(gas_name_cpp, gas_conc_1d)

            elif len(gas_conc.shape) == 2:
                dims_2d[0] = gas_conc.shape[1]
                dims_2d[1] = gas_conc.shape[0]

                gas_conc_2d.set_dims(dims_2d)

                # Adjust for the Fortran indexing used in Array.
                for ilev in range(gas_conc.shape[0]):
                    for icol in range(gas_conc.shape[1]):
                        index_2d[0] = icol+1
                        index_2d[1] = ilev+1
                        gas_conc_ptr = &gas_conc_2d(index_2d)
                        gas_conc_ptr[0] = gas_conc[ilev, icol]

                self.rad.set_vmr(gas_name_cpp, gas_conc_2d)

            else:
                raise RuntimeError('Illegal shape dimension')
        
        else:
            raise RuntimeError('Illegal input')


    def load_kdistribution_lw(self):
        cdef std_string file_name_cpp = b'coefficients_lw.nc'
        self.rad.load_kdistribution_lw(file_name_cpp)

