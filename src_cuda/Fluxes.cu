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

#include "Fluxes.h"
#include "Array.h"
#include "Optical_props.h"

namespace
{
    template<typename Float>__global__
    void sum_broadband(
                const int ncol, const int nlev, const int ngpt,
                const Float* __restrict__ spectral_flux, Float* __restrict__ broadband_flux)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilev = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol < ncol) && (ilev < nlev) )
        {
            const int idx_out = icol + ilev*ncol;
            Float bb_flux_s = 0;
            for (int igpt=0; igpt < ngpt; ++igpt)
            {
                const int idx_in = icol + ilev*ncol + igpt*nlev*ncol;
                bb_flux_s += spectral_flux[idx_in];
            }
            broadband_flux[idx_out] = bb_flux_s;
        }
    }

    template<typename Float>__global__
    void net_broadband_precalc(
                const int ncol, const int nlev,
                const Float* __restrict__ flux_dn, const Float* __restrict__ flux_up,
                Float* __restrict__ broadband_flux_net)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilev = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol < ncol) && (ilev < nlev) )
        {
            const int idx = icol + ilev*ncol;
            broadband_flux_net[idx] = flux_dn[idx] - flux_up[idx];
        }
    }

    template<typename Float>__global__
    void sum_byband(
                const int ncol, const int nlev, const int ngpt, const int nbnd,
                const int* __restrict__ band_lims, const Float* __restrict__ spectral_flux,
                Float* __restrict__ byband_flux)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilev = blockIdx.y*blockDim.y + threadIdx.y;
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( icol < ncol) && (ilev < nlev) && (ibnd < nbnd) )
        {
            const int idx_bnd = icol + ilev*ncol + ibnd*ncol*nlev;
            const int gpt_start = band_lims[2*ibnd];
            const int gpt_end = band_lims[2*ibnd+1];

            byband_flux[idx_bnd] = 0;

            for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
            {
                const int idx_gpt = icol + ilev*ncol + igpt*ncol*nlev;
                byband_flux[idx_bnd] += spectral_flux[idx_gpt];
            }
        }
    }

    template<typename Float>__global__
    void net_byband_full(
                const int ncol, const int nlev, const int ngpt, const int nbnd,
                const int* __restrict__ band_lims, const Float* __restrict__ spectral_flux_dn,
                const Float* __restrict__ spectral_flux_up, Float* __restrict__ byband_flux_net)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilev = blockIdx.y*blockDim.y + threadIdx.y;
        const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( icol < ncol) && (ilev < nlev) && (ibnd < nbnd) )
        {
            const int idx_bnd = icol + ilev*ncol + ibnd*ncol*nlev;
            const int gpt_start = band_lims[2*ibnd];
            const int gpt_end = band_lims[2*ibnd+1];
            byband_flux_net[idx_bnd] = 0;
            for (int igpt = gpt_start; igpt < gpt_end; ++igpt)
            {
                const int idx_gpt = icol + ilev*ncol + igpt*ncol*nlev;
                byband_flux_net[idx_bnd] += spectral_flux_dn[idx_gpt] - spectral_flux_up[idx_gpt];
            }
        }
    }
}

//namespace rrtmgp_kernel_launcher
//{
//    template<typename Float>
//    void sum_broadband(
//            int ncol, int nlev, int ngpt,
//            const Array<Float,3>& spectral_flux, Array<Float,2>& broadband_flux)
//    {
//        rrtmgp_kernels::sum_broadband(
//                &ncol, &nlev, &ngpt,
//                const_cast<Float*>(spectral_flux.ptr()),
//                broadband_flux.ptr());
//    }
//
//    template<typename Float>
//    void net_broadband(
//            int ncol, int nlev,
//            const Array<Float,2>& broadband_flux_dn, const Array<Float,2>& broadband_flux_up,
//            Array<Float,2>& broadband_flux_net)
//    {
//        rrtmgp_kernels::net_broadband_precalc(
//                &ncol, &nlev,
//                const_cast<Float*>(broadband_flux_dn.ptr()),
//                const_cast<Float*>(broadband_flux_up.ptr()),
//                broadband_flux_net.ptr());
//    }
//
//    template<typename Float>
//    void sum_byband(
//            int ncol, int nlev, int ngpt, int nbnd,
//            const Array<int,2>& band_lims,
//            const Array<Float,3>& spectral_flux,
//            Array<Float,3>& byband_flux)
//    {
//        rrtmgp_kernels::sum_byband(
//                &ncol, &nlev, &ngpt, &nbnd,
//                const_cast<int*>(band_lims.ptr()),
//                const_cast<Float*>(spectral_flux.ptr()),
//                byband_flux.ptr());
//    }
//
//    template<typename Float>
//    void net_byband(
//            int ncol, int nlev, int nband,
//            const Array<Float,3>& byband_flux_dn, const Array<Float,3>& byband_flux_up,
//            Array<Float,3>& byband_flux_net)
//    {
//        rrtmgp_kernels::net_byband_precalc(
//                &ncol, &nlev, &nband,
//                const_cast<Float*>(byband_flux_dn.ptr()),
//                const_cast<Float*>(byband_flux_up.ptr()),
//                byband_flux_net.ptr());
//    }

Fluxes_broadband_gpu::Fluxes_broadband_gpu(const int ncol, const int nlev) :
    flux_up    ({ncol, nlev}),
    flux_dn    ({ncol, nlev}),
    flux_dn_dir({ncol, nlev}),
    flux_net   ({ncol, nlev})
{}


void Fluxes_broadband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up, const Array_gpu<Float,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = gpt_flux_up.dim(3);

    const int block_lev = 16;
    const int block_col = 16;

    const int grid_col = ncol/block_col + (ncol%block_col > 0);
    const int grid_lev = nlev/block_lev + (nlev%block_lev > 0);

    dim3 grid_gpu(grid_col, grid_lev);
    dim3 block_gpu(block_col, block_lev);

    sum_broadband<<<grid_gpu, block_gpu>>>(
            ncol, nlev, ngpt, gpt_flux_up.ptr(), this->flux_up.ptr());

    sum_broadband<<<grid_gpu, block_gpu>>>(
            ncol, nlev, ngpt, gpt_flux_dn.ptr(), this->flux_dn.ptr());

    net_broadband_precalc<<<grid_gpu, block_gpu>>>(
            ncol, nlev, this->flux_dn.ptr(), this->flux_up.ptr(), this->flux_net.ptr());
}


//// CvH: unnecessary code duplication.
void Fluxes_broadband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up, const Array_gpu<Float,3>& gpt_flux_dn, const Array_gpu<Float,3>& gpt_flux_dn_dir,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = gpt_flux_up.dim(3);

    reduce(gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1);

    const int block_lev = 16;
    const int block_col = 16;

    const int grid_col = ncol/block_col + (ncol%block_col > 0);
    const int grid_lev = nlev/block_lev + (nlev%block_lev > 0);

    dim3 grid_gpu(grid_col, grid_lev);
    dim3 block_gpu(block_col, block_lev);

    sum_broadband<<<grid_gpu, block_gpu>>>(
            ncol, nlev, ngpt, gpt_flux_dn_dir.ptr(), this->flux_dn_dir.ptr());
}


Fluxes_byband_gpu::Fluxes_byband_gpu(const int ncol, const int nlev, const int nbnd) :
    Fluxes_broadband_gpu(ncol, nlev),
    bnd_flux_up    ({ncol, nlev, nbnd}),
    bnd_flux_dn    ({ncol, nlev, nbnd}),
    bnd_flux_dn_dir({ncol, nlev, nbnd}),
    bnd_flux_net   ({ncol, nlev, nbnd})
{}


void Fluxes_byband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up,
    const Array_gpu<Float,3>& gpt_flux_dn,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();
    const int nbnd = spectral_disc->get_nband();

    const Array_gpu<int,2>& band_lims = spectral_disc->get_band_lims_gpoint();
    const int block_bnd = 1;
    const int block_lev = 16;
    const int block_col = 16;

    const int grid_col = ncol/block_col + (ncol%block_col > 0);
    const int grid_lev = nlev/block_lev + (nlev%block_lev > 0);
    const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

    dim3 grid_gpu(grid_col, grid_lev, grid_bnd);
    dim3 block_gpu(block_col, block_lev, grid_bnd);

    Fluxes_broadband_gpu::reduce(
            gpt_flux_up, gpt_flux_dn,
            spectral_disc, top_at_1);

    sum_byband<<<grid_gpu, block_gpu>>>(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            gpt_flux_up.ptr(), this->bnd_flux_up.ptr());

    sum_byband<<<grid_gpu, block_gpu>>>(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            gpt_flux_dn.ptr(), this->bnd_flux_dn.ptr());

    net_byband_full<<<grid_gpu, block_gpu>>>(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            this->bnd_flux_dn.ptr(), this->bnd_flux_up.ptr(), this->bnd_flux_net.ptr());
}


// CvH: a lot of code duplication.
void Fluxes_byband_gpu::reduce(
    const Array_gpu<Float,3>& gpt_flux_up,
    const Array_gpu<Float,3>& gpt_flux_dn,
    const Array_gpu<Float,3>& gpt_flux_dn_dir,
    const std::unique_ptr<Optical_props_arry_gpu>& spectral_disc,
    const Bool top_at_1)
{
    const int ncol = gpt_flux_up.dim(1);
    const int nlev = gpt_flux_up.dim(2);
    const int ngpt = spectral_disc->get_ngpt();
    const int nbnd = spectral_disc->get_nband();

    const Array_gpu<int,2>& band_lims = spectral_disc->get_band_lims_gpoint();

    Fluxes_broadband_gpu::reduce(
            gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir,
            spectral_disc, top_at_1);

    reduce(gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1);

    const int block_bnd = 1;
    const int block_lev = 16;
    const int block_col = 16;

    const int grid_col = ncol/block_col + (ncol%block_col > 0);
    const int grid_lev = nlev/block_lev + (nlev%block_lev > 0);
    const int grid_bnd = nbnd/block_bnd + (nbnd%block_bnd > 0);

    dim3 grid_gpu(grid_col, grid_lev, grid_bnd);
    dim3 block_gpu(block_col, block_lev, grid_bnd);

    sum_byband<<<grid_gpu, block_gpu>>>(
            ncol, nlev, ngpt, nbnd, band_lims.ptr(),
            gpt_flux_dn_dir.ptr(), this->bnd_flux_dn_dir.ptr());
}
