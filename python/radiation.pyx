# distutils: language = c++
import cython
from libcpp.string cimport string as std_string
from libcpp cimport bool
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
        T* ptr()


cdef extern from "../include_test/Radiation_solver.h":
    cdef cppclass Radiation_solver[TF]:
        Radiation_solver() except +
        void load_kdistribution_longwave(const std_string&);
        void set_vmr(const std_string&, const TF);
        void set_vmr(const std_string&, const Array[TF,d1]&);
        void set_vmr(const std_string&, const Array[TF,d2]&);
        void solve_longwave(
                const bool sw_output_optical,
                const bool sw_output_bnd_fluxes,
                const Array[TF,d2]& p_lay, const Array[TF,d2]& p_lev,
                const Array[TF,d2]& t_lay, const Array[TF,d2]& t_lev,
                # const Array[TF,d2]& col_dry,
                const Array[TF,d1]& t_sfc, const Array[TF,d2]& emis_sfc,
                Array[TF,d3]& tau, Array[TF,d3]& lay_source,
                Array[TF,d3]& lev_source_inc, Array[TF,d3]& lev_source_dec, Array[TF,d2]& sfc_source,
                Array[TF,d2]& lw_flux_up, Array[TF,d2]& lw_flux_dn, Array[TF,d2]& lw_flux_net,
                Array[TF,d3]& lw_bnd_flux_up, Array[TF,d3]& lw_bnd_flux_dn, Array[TF,d3]& lw_bnd_flux_net);

# ctypedef fused float_t:
#     cython.float
#     cython.double

cdef copy_raw(double* a_in, double* a_out, int n):
    for i in range(n):
        a_out[i] = a_in[i]


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


    def load_kdistribution_longwave(self):
        cdef std_string file_name_cpp = b'coefficients_lw.nc'
        self.rad.load_kdistribution_longwave(file_name_cpp)



    def solve_longwave(self,
                sw_output_optical,
                sw_output_bnd_fluxes,
                np.ndarray[double, ndim=2, mode="c"] p_lay not None,
                np.ndarray[double, ndim=2, mode="c"] p_lev not None,
                np.ndarray[double, ndim=2, mode="c"] t_lay not None,
                np.ndarray[double, ndim=2, mode="c"] t_lev not None,
                # np.ndarray[double, ndim=2, mode="c"] col_dry not None,
                np.ndarray[double, ndim=1, mode="c"] t_sfc not None,
                np.ndarray[double, ndim=2, mode="c"] emis_sfc not None,
                np.ndarray[double, ndim=3, mode="c"] tau not None,
                np.ndarray[double, ndim=3, mode="c"] lay_source not None,
                np.ndarray[double, ndim=3, mode="c"] lev_source_inc not None,
                np.ndarray[double, ndim=3, mode="c"] lev_source_dec not None,
                np.ndarray[double, ndim=2, mode="c"] sfc_source not None,
                np.ndarray[double, ndim=2, mode="c"] lw_flux_up not None,
                np.ndarray[double, ndim=2, mode="c"] lw_flux_dn not None,
                np.ndarray[double, ndim=2, mode="c"] lw_flux_net not None,
                np.ndarray[double, ndim=3, mode="c"] lw_bnd_flux_up not None,
                np.ndarray[double, ndim=3, mode="c"] lw_bnd_flux_dn not None,
                np.ndarray[double, ndim=3, mode="c"] lw_bnd_flux_net not None):

        ncol = p_lay.shape[1]
        nlay = p_lay.shape[0]
        nlev = p_lev.shape[0]

        nbnd = emis_sfc.shape[1]

        cdef Array[double,d2] p_lay_cpp, p_lev_cpp, t_lay_cpp, t_lev_cpp # , col_dry_cpp
        cdef Array[double,d1] t_sfc_cpp,
        cdef Array[double,d2] emis_sfc_cpp
        cdef Array[double,d3] tau_cpp, lay_source_cpp, lev_source_inc_cpp, lev_source_dec_cpp
        cdef Array[double,d2] sfc_source_cpp
        cdef Array[double,d2] lw_flux_up_cpp, lw_flux_dn_cpp, lw_flux_net_cpp
        cdef Array[double,d3] lw_bnd_flux_up_cpp, lw_bnd_flux_dn_cpp, lw_bnd_flux_net_cpp

        cdef array[int,d2] d_ncol_nlay
        d_ncol_nlay[0], d_ncol_nlay[1] = ncol, nlay

        cdef array[int,d2] d_ncol_nlev
        d_ncol_nlev[0], d_ncol_nlev[1] = ncol, nlev

        cdef array[int,d1] d_ncol
        d_ncol[0] = ncol

        cdef array[int,d2] d_nbnd_ncol
        d_nbnd_ncol[0], d_nbnd_ncol[1] = nbnd, ncol

        p_lay_cpp.set_dims(d_ncol_nlay)
        p_lev_cpp.set_dims(d_ncol_nlev)
        t_lay_cpp.set_dims(d_ncol_nlay)
        t_lev_cpp.set_dims(d_ncol_nlev)
        # col_dry_cpp.set_dims(d_ncol_nlay)
        t_sfc_cpp.set_dims(d_ncol)
        emis_sfc_cpp.set_dims(d_nbnd_ncol)
        lw_flux_up_cpp.set_dims(d_ncol_nlev)
        lw_flux_dn_cpp.set_dims(d_ncol_nlev)
        lw_flux_net_cpp.set_dims(d_ncol_nlev)

        copy_raw(&p_lay[0,0], p_lay_cpp.ptr(), p_lay.size)
        copy_raw(&p_lev[0,0], p_lev_cpp.ptr(), p_lev.size)
        copy_raw(&t_lay[0,0], t_lay_cpp.ptr(), t_lay.size)
        copy_raw(&t_lev[0,0], t_lev_cpp.ptr(), t_lev.size)
        # copy_raw(&col_dry[0,0], col_dry_cpp.ptr(), col_dry.size)
        copy_raw(&t_sfc[0], t_sfc_cpp.ptr(), t_sfc.size)
        copy_raw(&emis_sfc[0,0], emis_sfc_cpp.ptr(), emis_sfc.size)

        self.rad.solve_longwave(
                sw_output_optical,
                sw_output_bnd_fluxes,
                p_lay_cpp, p_lev_cpp,
                t_lay_cpp, t_lev_cpp,
                # col_dry_cpp,
                t_sfc_cpp, emis_sfc_cpp,
                tau_cpp, lay_source_cpp,
                lev_source_inc_cpp, lev_source_dec_cpp, sfc_source_cpp,
                lw_flux_up_cpp, lw_flux_dn_cpp, lw_flux_net_cpp,
                lw_bnd_flux_up_cpp, lw_bnd_flux_dn_cpp, lw_bnd_flux_net_cpp)

        copy_raw(lw_flux_up_cpp .ptr(), &lw_flux_up [0,0], lw_flux_up .size)
        copy_raw(lw_flux_dn_cpp .ptr(), &lw_flux_dn [0,0], lw_flux_dn .size)
        copy_raw(lw_flux_net_cpp.ptr(), &lw_flux_net[0,0], lw_flux_net.size)

