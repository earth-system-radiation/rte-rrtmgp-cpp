#include "rrtmgp_kernels_cuda.h"

namespace rrtmgp_kernels_cuda
{
    template<typename TF>
    void combine_and_reorder_2str(
            const int ncol, const int nlay, const int ngpt,
            const TF* tau_local, const TF* tau_rayleigh,
            TF* tau, TF* ssa, TF* g)
    {}
}

#ifdef FLOAT_SINGLE_RRTMGP
template void rrtmgp_kernels_cuda::combine_and_reorder_2str<float>(
        const int, const int, const int, const float*, const float*, float*, float*, float*);
#else
template void rrtmgp_kernels_cuda::combine_and_reorder_2str<double>(
        const int, const int, const int, const double*, const double*, double*, double*, double*);
#endif
