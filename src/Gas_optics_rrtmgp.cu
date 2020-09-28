/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#include "Gas_concs.h"
#include "Gas_optics_rrtmgp.h"
#include "Array.h"
#include "iostream"

template<typename TF>__global__
void fill_gases_kernel(
        const int ncol, const int nlay, const int ngas, const int igas, 
        TF* __restrict__ vmr_out, TF* __restrict__ vmr_in,
        TF* __restrict__ col_gas, const TF* __restrict__ col_dry)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    if ( ( icol < ncol) && (ilay < nlay) )
    {
        const int idx_in = icol + ilay*ncol;
        const int idx_out = icol + ilay*ncol;
        if (igas > 0)
        {
            if (ncol == 1 && nlay == 1)
            { 
                 vmr_out[idx_out] = vmr_in[0];
            }
            else if (ncol == 1)
            {
                 vmr_out[idx_out] = vmr_in[ilay];
            }
            else
            {
                vmr_out[idx_out] = vmr_in[idx_in];
            }
            col_gas[idx_out] = vmr_out[idx_out] * col_dry[idx_in];
        }
        else if (igas == 0)
        {
            col_gas[idx_out] = col_dry[idx_in];
        }
    }
}

template<typename TF> __global__
void compute_delta_plev(
        const int ncol, const int nlay,
        const TF* __restrict__ plev,
        TF* __restrict__ delta_plev)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        delta_plev[idx] = abs(plev[idx] - plev[idx + ncol]);

        // delta_plev({icol, ilay}) = std::abs(plev({icol, ilay}) - plev({icol, ilay+1}));
    }
}

template<typename TF> __global__
void compute_m_air(
        const int ncol, const int nlay,
        const TF* __restrict__ vmr_h2o,
        TF* __restrict__ m_air)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr TF m_dry = 0.028964;
    constexpr TF m_h2o = 0.018016;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        m_air[idx] = (m_dry + m_h2o * vmr_h2o[idx]) / (TF(1.) + vmr_h2o[idx]);

        // m_air({icol, ilay}) = (m_dry + m_h2o * vmr_h2o({icol, ilay})) / (1. + vmr_h2o({icol, ilay}));
    }
}

template<typename TF> __global__
void compute_col_dry(
        const int ncol, const int nlay,
        const TF* __restrict__ delta_plev, const TF* __restrict__ m_air, const TF* __restrict__ vmr_h2o,
        TF* __restrict__ col_dry)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr TF g0 = 9.80665;
    constexpr TF avogad = 6.02214076e23;

    if ( (icol < ncol) && (ilay < nlay) )
    {
        const int idx = icol + ilay*ncol;

        col_dry[idx] = TF(10.) * delta_plev[idx] * avogad / (TF(1000.)*m_air[idx]*TF(100.)*g0);
        col_dry[idx] /= (TF(1.) + vmr_h2o[idx]);

        // col_dry({icol, ilay}) = TF(10.) * delta_plev({icol, ilay}) * avogad / (TF(1000.)*m_air({icol, ilay})*TF(100.)*g0);
        // col_dry({icol, ilay}) /= (TF(1.) + vmr_h2o({icol, ilay}));
    }
}

// Calculate the molecules of dry air.
template<typename TF>
void Gas_optics_rrtmgp<TF>::get_col_dry_gpu(
        Array_gpu<TF,2>& col_dry, const Array_gpu<TF,2>& vmr_h2o,
        const Array_gpu<TF,2>& plev)
{
    Array_gpu<TF,2> delta_plev({col_dry.dim(1), col_dry.dim(2)});
    Array_gpu<TF,2> m_air     ({col_dry.dim(1), col_dry.dim(2)});

    const int block_lay = 16;
    const int block_col = 16;

    const int nlay = col_dry.dim(2);
    const int ncol = col_dry.dim(1);

    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    compute_delta_plev<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            plev.ptr(),
            delta_plev.ptr());

    compute_m_air<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            vmr_h2o.ptr(),
            m_air.ptr());

    compute_col_dry<<<grid_gpu, block_gpu>>>(
            ncol, nlay,
            delta_plev.ptr(), m_air.ptr(), vmr_h2o.ptr(),
            col_dry.ptr());
}

template<typename TF>
void Gas_optics_rrtmgp<TF>::gas_optics_gpu(
        const Array_gpu<TF,2>& play,
        const Array_gpu<TF,2>& plev,
        const Array_gpu<TF,2>& tlay,
        const Gas_concs_gpu<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Array_gpu<TF,2>& toa_src,
        const Array_gpu<TF,2>& col_dry) const
{
    std::cout<<"?????"<<std::endl;
    const int ncol = play.dim(1);
    const int nlay = play.dim(2);
    const int ngpt = this->get_ngpt();
    const int nband = this->get_nband();

    Array_gpu<int,2> jtemp({play.dim(1), play.dim(2)});
    Array_gpu<int,2> jpress({play.dim(1), play.dim(2)});
    Array_gpu<BOOL_TYPE,2> tropo({play.dim(1), play.dim(2)});
    Array_gpu<TF,6> fmajor({2, 2, 2, this->get_nflav(), play.dim(1), play.dim(2)});
    Array_gpu<int,4> jeta({2, this->get_nflav(), play.dim(1), play.dim(2)});
    std::cout<<"?????"<<std::endl;
    // Gas optics.
    compute_gas_taus_gpu(
            ncol, nlay, ngpt, nband,
            play, plev, tlay, gas_desc,
            optical_props,
            jtemp, jpress, jeta, tropo, fmajor,
            col_dry);

    // External source function is constant.
    for (int igpt=1; igpt<=ngpt; ++igpt)
        for (int icol=1; icol<=ncol; ++icol)
            toa_src.insert({icol, igpt},this->solar_source({igpt}));
}


template<typename TF>
void Gas_optics_rrtmgp<TF>::compute_gas_taus_gpu(
        const int ncol, const int nlay, const int ngpt, const int nband,
        const Array_gpu<TF,2>& play,
        const Array_gpu<TF,2>& plev,
        const Array_gpu<TF,2>& tlay,
        const Gas_concs_gpu<TF>& gas_desc,
        std::unique_ptr<Optical_props_arry<TF>>& optical_props,
        Array_gpu<int,2>& jtemp, Array_gpu<int,2>& jpress,
        Array_gpu<int,4>& jeta,
        Array_gpu<BOOL_TYPE,2>& tropo,
        Array_gpu<TF,6>& fmajor,
        const Array_gpu<TF,2>& col_dry) const
{
    Array_gpu<TF,3> tau({ngpt, nlay, ncol});
    Array_gpu<TF,3> tau_rayleigh({ngpt, nlay, ncol});
    Array_gpu<TF,3> vmr({ncol, nlay, this->get_ngas()});
    Array_gpu<TF,3> col_gas({ncol, nlay, this->get_ngas()+1});
    col_gas.set_offsets({0, 0, -1});
    Array_gpu<TF,4> col_mix({2, this->get_nflav(), ncol, nlay});
    Array_gpu<TF,5> fminor({2, 2, this->get_nflav(), ncol, nlay});

    // CvH add all the checking...
    const int ngas = this->get_ngas();
    const int nflav = this->get_nflav();
    const int neta = this->get_neta();
    const int npres = this->get_npres();
    const int ntemp = this->get_ntemp();

    const int nminorlower = this->minor_scales_with_density_lower.dim(1);
    const int nminorklower = this->kminor_lower.dim(1);
    const int nminorupper = this->minor_scales_with_density_upper.dim(1);
    const int nminorkupper = this->kminor_upper.dim(1);
    
    const int block_lay = 16;
    const int block_col = 16;

    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);
   
    for (int igas=0; igas<=ngas; ++igas)
    {
        Array_gpu<TF,2> vmr_2d = gas_desc.get_vmr(this->gas_names({igas}));
        fill_gases_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ngas, igas, vmr.ptr(), vmr_2d.ptr(), col_gas.ptr(), col_dry.ptr());
    }
    vmr.dump("vmr_subset_gpu");
    throw 666;
} 








#ifdef FLOAT_SINGLE_RRTMGP
template class Gas_optics_rrtmgp<float>;
#else
template class Gas_optics_rrtmgp<double>;
#endif
