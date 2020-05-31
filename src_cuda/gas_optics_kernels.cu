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
