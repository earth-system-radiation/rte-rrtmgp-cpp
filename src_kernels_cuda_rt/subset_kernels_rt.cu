__global__
void get_from_subset_kernel(const int ncol, const int nbnd, const int ncol_in, const int col_s_in,
              Float* __restrict__ var_full, const Float* __restrict__ var_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ibnd = blockIdx.y*blockDim.y + threadIdx.y;
    if ( (icol < ncol_in) && (ibnd < nbnd) )
    {
        const int idx_full = icol+col_s_in-1 + ibnd*ncol;
        const int idx_sub = icol + ibnd*ncol_in;
        var_full[idx_full] = var_sub[idx_sub];
    }
}

__global__
void get_from_subset_kernel(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
              Float* __restrict__ var1_full, Float* __restrict__ var2_full, Float* __restrict__ var3_full,  Float* __restrict__ var4_full,
              const Float* __restrict__ var1_sub, const Float* __restrict__ var2_sub, const Float* __restrict__ var3_sub, const Float* __restrict__ var4_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol_in) && (ilay < nlay) )
    {
        const int idx_full = icol+col_s_in-1 + ilay*ncol;
        const int idx_sub = icol + ilay*ncol_in;
        var1_full[idx_full] = var1_sub[idx_sub];
        var2_full[idx_full] = var2_sub[idx_sub];
        var3_full[idx_full] = var3_sub[idx_sub];
        var4_full[idx_full] = var4_sub[idx_sub];
    }
}

__global__
void get_from_subset_kernel(const int ncol, const int nlay, const int ncol_in, const int col_s_in,
              Float* __restrict__ var1_full, Float* __restrict__ var2_full, Float* __restrict__ var3_full,
              const Float* __restrict__ var1_sub, const Float* __restrict__ var2_sub, const Float* __restrict__ var3_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

    if ( (icol < ncol_in) && (ilay < nlay) )
    {
        const int idx_full = icol+col_s_in-1 + ilay*ncol;
        const int idx_sub = icol + ilay*ncol_in;
        var1_full[idx_full] = var1_sub[idx_sub];
        var2_full[idx_full] = var2_sub[idx_sub];
        var3_full[idx_full] = var3_sub[idx_sub];
    }
}

__global__
void get_from_subset_kernel(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
              Float* __restrict__ var1_full, Float* __restrict__ var2_full, Float* __restrict__ var3_full,  Float* __restrict__ var4_full,
              const Float* __restrict__ var1_sub, const Float* __restrict__ var2_sub, const Float* __restrict__ var3_sub, const Float* __restrict__ var4_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol_in) && (ilay < nlay) && (ibnd < nbnd) )
    {
        const int idx_full = icol+col_s_in-1 + ilay*ncol + ibnd*nlay*ncol;
        const int idx_sub = icol + ilay*ncol_in + ibnd*nlay*ncol_in;
        var1_full[idx_full] = var1_sub[idx_sub];
        var2_full[idx_full] = var2_sub[idx_sub];
        var3_full[idx_full] = var3_sub[idx_sub];
        var4_full[idx_full] = var4_sub[idx_sub];
    }
}

__global__
void get_from_subset_kernel(const int ncol, const int nlay, const int nbnd, const int ncol_in, const int col_s_in,
              Float* __restrict__ var1_full, Float* __restrict__ var2_full, Float* __restrict__ var3_full,
              const Float* __restrict__ var1_sub, const Float* __restrict__ var2_sub, const Float* __restrict__ var3_sub)
{
    const int icol = blockIdx.x*blockDim.x + threadIdx.x;
    const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
    const int ibnd = blockIdx.z*blockDim.z + threadIdx.z;

    if ( (icol < ncol_in) && (ilay < nlay) && (ibnd < nbnd) )
    {
        const int idx_full = icol+col_s_in-1 + ilay*ncol + ibnd*nlay*ncol;
        const int idx_sub = icol + ilay*ncol_in + ibnd*nlay*ncol_in;
        var1_full[idx_full] = var1_sub[idx_sub];
        var2_full[idx_full] = var2_sub[idx_sub];
        var3_full[idx_full] = var3_sub[idx_sub];
    }
}
