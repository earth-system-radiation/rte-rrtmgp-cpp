#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"

#include "Array.h"
#include "chrono"

namespace
{
    // Add the kernels here.
    template<typename TF>__device__
    void interpolate2D_byflav_kernel(const TF* __restrict__ fminor,
                                     const TF* __restrict__ krayl,
                                     const int gptS, const int gptE,
                                     TF* __restrict__ k,
                                     const int* __restrict__ jeta,
                                     const int jtemp)
    {
        const int ngpt = gptE-gptS;
        const int jeta_size = 2;
        for (int igpt=gptS; igpt<gptE; ++igpt)
        {
            k[igpt-gptS] = fminor[0] * krayl[igpt + jeta[0]*ngpt     + jtemp    *jeta_size*ngpt] +
                           fminor[1] * krayl[igpt + (jeta[0]+1)*ngpt + jtemp    *jeta_size*ngpt] +
                           fminor[2] * krayl[igpt + jeta[1]*ngpt     + (jtemp+1)*jeta_size*ngpt] +
                           fminor[3] * krayl[igpt + (jeta[1]+1)*ngpt + (jtemp+1)*jeta_size*ngpt]; 
        }
    }

    template<typename TF>__global__
    void compute_tau_rayleigh_kernel(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* __restrict__ gpoint_flavor,
            const int* __restrict__ band_lims_gpt,
            const TF* __restrict__ krayl,
            int idx_h2o, const TF* __restrict__ col_dry, const TF* __restrict__ col_gas,
            const TF* __restrict__ fminor, const int* __restrict__ jeta,
            const BOOL_TYPE* __restrict__ tropo, const int* __restrict__ jtemp,
            TF* __restrict__ tau_rayleigh)
    {
        // Fetch the three coordinates.
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ibnd = blockIdx.y*blockDim.y + threadIdx.y;
        const int ilay = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (ibnd < nbnd) )
        {
            //kernel implementation
            const int idx_collay = icol + ilay*ncol;
            const int idx_collaywv = icol + ilay*ncol + (idx_h2o-1)*nlay*ncol;
            const int itropo = tropo[idx_collay];
            const int gptS = band_lims_gpt[ibnd]-1;
            const int gptE = band_lims_gpt[ibnd+nbnd];
            const int iflav = gpoint_flavor[itropo+2*gptS]-1;
            const int idx_fminor = 2*2*(iflav + icol*nflav + ilay*ncol*nflav);
            const int idx_jeta   = 2*(iflav + icol*nflav + ilay*ncol*nflav);
            const int idx_krayl  = gptS+ngpt*neta*ntemp*itropo;

            TF k[ngpt];
            interpolate2D_byflav_kernel(&fminor[idx_fminor],
                                        &krayl[idx_krayl],
                                        gptS, gptE, &k,
                                        &jeta[idx_jeta],
                                        jtemp[idx_collay]);
            for (int igpt=gptS; igpt<gptE; ++igpt)
            {
                const int idx_out = igpt + ilay*ngpt + icol*nlay*ngpt;
                tau_rayleigh[idx_out] = k[igpt]*(col_gas[idx_collaywv]+col_dry[idx_collay]);
            }
        }
    }

    template<typename TF>__global__
    void combine_and_reorder_2str_kernel(
            const int ncol, const int nlay, const int ngpt, const TF tmin,
            const TF* __restrict__ tau_abs, const TF* __restrict__ tau_rayleigh,
            TF* __restrict__ tau, TF* __restrict__ ssa, TF* __restrict__ g)
    {
        // Fetch the three coordinates.
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int igpt = blockIdx.y*blockDim.y + threadIdx.y;
        const int ilay = blockIdx.z*blockDim.z + threadIdx.z;

        if ( (icol < ncol) && (ilay < nlay) && (igpt < ngpt) )
        {
            const int idx_in  = igpt + ilay*ngpt + icol*(ngpt*nlay);
            const int idx_out = icol + ilay*ncol + igpt*(ncol*nlay);
	   
	    const TF tau_tot = tau_abs[idx_in] + tau_rayleigh[idx_in];
	    tau[idx_out] = tau_tot;
            g  [idx_out] = TF(0.);
            if (tau_tot>(TF(2.)*tmin))
                ssa[idx_out] = tau_rayleigh[idx_in]/tau_tot;
            else
                ssa[idx_out] = 0.;
		
        }
    }
}

namespace rrtmgp_kernel_launcher_cuda
{
    template<typename TF>
    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const Array<TF,3>& tau_abs, const Array<TF,3>& tau_rayleigh,
            Array<TF,3>& tau, Array<TF,3>& ssa, Array<TF,3>& g)
    {
        // Store the sizes, all the same
        const int array_size = tau_abs.size()*sizeof(TF);

        TF* tau_abs_gpu;
        TF* tau_rayleigh_gpu;
        TF* tau_gpu;
        TF* ssa_gpu;
        TF* g_gpu;
        TF tmin = std::numeric_limits<TF>::min();
        // Allocate a CUDA array.
        cuda_safe_call(cudaMalloc((void**)&tau_abs_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&tau_rayleigh_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&tau_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&ssa_gpu, array_size));
        cuda_safe_call(cudaMalloc((void**)&g_gpu, array_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(tau_abs_gpu, tau_abs.ptr(), array_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tau_rayleigh_gpu, tau_rayleigh.ptr(), array_size, cudaMemcpyHostToDevice));
        cudaEvent_t startEvent, stopEvent;
        float dt1;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent, 0);
        // Call the kernel.
        const int block_col = 32;
        const int block_gpt = 32;
        const int block_lay = 1;

        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_gpt  = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_gpt, grid_lay);
        dim3 block_gpu(block_col, block_gpt, block_lay);

        combine_and_reorder_2str_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs_gpu, tau_rayleigh_gpu,
                tau_gpu, ssa_gpu, g_gpu);
        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&dt1,startEvent,stopEvent);

//        std::cout<<"GPU kernel "<<dt1<<" (ms)"<<std::endl;

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(tau.ptr(), tau_gpu, array_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(ssa.ptr(), ssa_gpu, array_size, cudaMemcpyDeviceToHost));
        cuda_safe_call(cudaMemcpy(g.ptr(), g_gpu, array_size, cudaMemcpyDeviceToHost));

        // Deallocate a CUDA array.
        cuda_safe_call(cudaFree(tau_abs_gpu));
        cuda_safe_call(cudaFree(tau_rayleigh_gpu));
        cuda_safe_call(cudaFree(tau_gpu));
        cuda_safe_call(cudaFree(ssa_gpu));
        cuda_safe_call(cudaFree(g_gpu));
    }
    
    template<typename TF>
    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const Array<int, 2>& gpoint_flavor,
            const Array<int, 2>& band_lims_gpt,
            const Array<TF,4>& krayl,
            int idx_h2o, const Array<TF,2>& col_dry, const Array<TF,3>& col_gas,
            const Array<TF,5>& fminor, const Array<int,4>& jeta,
            const Array<BOOL_TYPE,2>& tropo, const Array<int,2>& jtemp,
            Array<TF,3>& tau_rayleigh)
    {
        float elapsedtime;
        const int gpoint_flavor_size = gpoint_flavor.size()*sizeof(int);
        const int band_lims_gpt_size = band_lims_gpt.size()*sizeof(int);
        const int krayl_size = krayl.size()*sizeof(TF);
        const int col_dry_size = col_dry.size()*sizeof(TF);
        const int col_gas_size = col_gas.size()*sizeof(TF);
        const int fminor_size = fminor.size()*sizeof(TF);
        const int jeta_size = jeta.size()*sizeof(int);
        const int tropo_size = tropo.size()*sizeof(BOOL_TYPE);
        const int jtemp_size = jtemp.size()*sizeof(int);
        const int tau_rayleigh_size = tau_rayleigh.size()*sizeof(TF);

        int* gpoint_flavor_gpu;
        int* band_lims_gpt_gpu;
        int* jeta_gpu;
        int* jtemp_gpu;
        BOOL_TYPE* tropo_gpu;
        TF* krayl_gpu;
        TF* col_dry_gpu;
        TF* col_gas_gpu;
        TF* fminor_gpu;
        TF* tau_rayleigh_gpu;
        TF tmin = std::numeric_limits<TF>::min();

        // Allocate a CUDA array.
        cuda_safe_call(cudaMalloc((void**)&gpoint_flavor_gpu, gpoint_flavor_size));
        cuda_safe_call(cudaMalloc((void**)&band_lims_gpt_gpu, band_lims_gpt_size));
        cuda_safe_call(cudaMalloc((void**)&krayl_gpu, krayl_size));
        cuda_safe_call(cudaMalloc((void**)&col_dry_gpu, col_dry_size));
        cuda_safe_call(cudaMalloc((void**)&col_gas_gpu, col_gas_size));
        cuda_safe_call(cudaMalloc((void**)&fminor_gpu, fminor_size));
        cuda_safe_call(cudaMalloc((void**)&jeta_gpu, jeta_size));
        cuda_safe_call(cudaMalloc((void**)&tropo_gpu, tropo_size));
        cuda_safe_call(cudaMalloc((void**)&jtemp_gpu, jtemp_size));
        cuda_safe_call(cudaMalloc((void**)&tau_rayleigh_gpu, tau_rayleigh_size));

        // Copy the data to the GPU.
        cuda_safe_call(cudaMemcpy(gpoint_flavor_gpu, gpoint_flavor.ptr(), gpoint_flavor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(band_lims_gpt_gpu, band_lims_gpt.ptr(), band_lims_gpt_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(krayl_gpu, krayl.ptr(), krayl_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(col_dry_gpu, col_dry.ptr(), col_dry_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(col_gas_gpu, col_gas.ptr(), col_gas_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(fminor_gpu, fminor.ptr(), fminor_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(jeta_gpu, jeta.ptr(), jeta_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tropo_gpu, tropo.ptr(), tropo_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(jtemp_gpu, jtemp.ptr(), jtemp_size, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpy(tau_rayleigh_gpu, tau_rayleigh.ptr(), tau_rayleigh_size, cudaMemcpyHostToDevice));

        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent, 0);

        // Call the kernel.
        const int block_col = 32;
        const int block_bnd = 1;
        const int block_lay = 1;

        const int grid_col  = ncol/block_col + (ncol%block_col > 0);
        const int grid_bnd  = nbnd/block_bnd + (nbnd%block_bnd > 0);
        const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

        dim3 grid_gpu(grid_col, grid_bnd, grid_lay);
        dim3 block_gpu(block_col, block_bnd, block_lay);

        compute_tau_rayleigh_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor_gpu,
                band_lims_gpt_gpu,
                krayl_gpu,
                idx_h2o, col_dry_gpu, col_gas_gpu,
                fminor_gpu, jeta_gpu,
                tropo_gpu, jtemp_gpu,
                tau_rayleigh_gpu);

        cuda_check_error();
        cuda_safe_call(cudaDeviceSynchronize());
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedtime,startEvent,stopEvent);
        std::cout<<"GPU kernel "<<elapsedtime<<" (ms)"<<std::endl;

        // Copy back the results.
        cuda_safe_call(cudaMemcpy(tau_rayleigh.ptr(), tau_rayleigh_gpu, tau_rayleigh_size, cudaMemcpyDeviceToHost));

        cuda_safe_call(cudaFree(gpoint_flavor_gpu));
        cuda_safe_call(cudaFree(band_lims_gpt_gpu));
        cuda_safe_call(cudaFree(krayl_gpu));
        cuda_safe_call(cudaFree(col_dry_gpu));
        cuda_safe_call(cudaFree(col_gas_gpu));
        cuda_safe_call(cudaFree(fminor_gpu));
        cuda_safe_call(cudaFree(jeta_gpu));
        cuda_safe_call(cudaFree(tropo_gpu));
        cuda_safe_call(cudaFree(jtemp_gpu));
        cuda_safe_call(cudaFree(tau_rayleigh_gpu));

    }
    
}

#ifdef FLOAT_SINGLE_RRTMGP
template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<float>(
        const int, const int, const int, const Array<float,3>&, const Array<float,3>&, Array<float,3>&, Array<float,3>&, Array<float,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_rayleigh<float>(
        const int, const int, const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<int,2>&, const Array<float,4>&, int, const Array<float,2>&, 
        const Array<float,3>&, const Array<float,5>&, const Array<int,4>&, const Array<BOOL_TYPE,2>&, 
        const Array<int,2>&, Array<float,3>&);


#else
template void rrtmgp_kernel_launcher_cuda::combine_and_reorder_2str<double>(
        const int, const int, const int, const Array<double,3>&, const Array<double,3>&, Array<double,3>&, Array<double,3>&, Array<double,3>&);

template void rrtmgp_kernel_launcher_cuda::compute_tau_rayleigh<double>(
        const int, const int, const int, const int, const int, const int, const int, const int, const int,
        const Array<int,2>&, const Array<int,2>&, const Array<double,4>&, int, const Array<double,2>&, 
        const Array<double,3>&, const Array<double,5>&, const Array<int,4>&, const Array<BOOL_TYPE,2>&, 
        const Array<int,2>&, Array<double,3>&);
#endif


