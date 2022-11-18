#include <curand_kernel.h>
#include "raytracer_kernels.h"
#include <iostream>

namespace
{
    using namespace Raytracer_functions;

    constexpr Float w_thres = 0.5;
    //constexpr Float solar_cone_cos_half_angle = Float(0.9961947); // cos(Float(5.0) / Float(180.) * M_PI;)
    constexpr Float solar_cone_cos_half_angle = Float(0.99904823); // cos(Float(2.5) / Float(180.) * M_PI;)

    struct Quasi_random_number_generator_2d
    {
        __device__ Quasi_random_number_generator_2d(
                curandDirectionVectors32_t* vectors, unsigned int* constants, unsigned int offset)
        {
            curand_init(vectors[0], constants[0], offset, &state_x);
            curand_init(vectors[1], constants[1], offset, &state_y);
        }

        __device__ void xy(unsigned int* x, unsigned int* y,
                           const int itot, const int jtot,
                           const Int qrng_grid_x, const Int qrng_grid_y,
                           Int& photons_shot)
        {
            *x = curand(&state_x);
            *y = curand(&state_y);

            while (true)
            {
                const int i = *x / static_cast<unsigned int>((1ULL << 32) / qrng_grid_x);
                const int j = *y / static_cast<unsigned int>((1ULL << 32) / qrng_grid_y);

                ++photons_shot;
                if (i < itot && j < jtot)
                {
                    return;
                }
                else
                {
                    *x = curand(&state_x);
                    *y = curand(&state_y);
                }
            }
        }

        curandStateScrambledSobol32_t state_x;
        curandStateScrambledSobol32_t state_y;
    };

    __device__
    inline void reset_photon(
            Photon& photon, Int& photons_shot, const Int photons_to_shoot,
            const Int qrng_grid_x, const Int qrng_grid_y,
            Float* __restrict__ const toa_down_count,
            Quasi_random_number_generator_2d& qrng,
            Random_number_generator<Float>& rng,
            const Float tod_inc_direct, const Float tod_inc_diffuse,
            const Float x_size, const Float y_size, const Float z_size,
            const Float dx_grid, const Float dy_grid, const Float dz_grid,
            const Float dir_x, const Float dir_y, const Float dir_z,
            Float& weight, const int itot, const int jtot)
    {
        unsigned int random_number_x;
        unsigned int random_number_y;
        qrng.xy(&random_number_x, &random_number_y, itot, jtot, qrng_grid_x, qrng_grid_y, photons_shot);

        if (photons_shot < photons_to_shoot)
        {
            const int i = random_number_x / static_cast<unsigned int>((1ULL << 32) / qrng_grid_x);
            const int j = random_number_y / static_cast<unsigned int>((1ULL << 32) / qrng_grid_y);

            photon.position.x = x_size * random_number_x / static_cast<unsigned int>((1ULL << 32) / qrng_grid_x) / itot;
            photon.position.y = y_size * random_number_y / static_cast<unsigned int>((1ULL << 32) / qrng_grid_y) / jtot;
            photon.position.z = z_size;

            const Float tod_diff_frac = tod_inc_diffuse / (tod_inc_direct + tod_inc_diffuse);
            if (rng() >= tod_diff_frac)
            {
                photon.direction.x = dir_x;
                photon.direction.y = dir_y;
                photon.direction.z = dir_z;
                photon.kind = Photon_kind::Direct;
            }
            else
            {
                const Float mu_surface = sqrt(rng());
                const Float azimuth_surface = Float(2.*M_PI)*rng();

                photon.direction.x = mu_surface*sin(azimuth_surface);
                photon.direction.y = mu_surface*cos(azimuth_surface);
                photon.direction.z = Float(-1) * (sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon));
                photon.kind = Photon_kind::Diffuse;
            }

            const int ij = i + j*itot;

            #ifndef NDEBUG
            if (ij < 0 || ij >=itot*jtot) printf("outofbounds 3");
            #endif

            atomicAdd(&toa_down_count[ij], Float(1.));
            weight = 1;

        }
    }

    template<typename T> __device__
    inline Float from_solar_cone(
        const Vector<T>& solar_dir,
        const Vector<T>& photon_dir)
    {
        return dot(solar_dir, photon_dir) > solar_cone_cos_half_angle;
    }
}


__global__
void ray_tracer_kernel(
        const Int photons_to_shoot,
        const Int qrng_grid_x,
        const Int qrng_grid_y,
        const Int qrng_gpt_offset,
        const Float* __restrict__ k_null_grid,
        Float* __restrict__ toa_down_count,
        Float* __restrict__ tod_up_count,
        Float* __restrict__ surface_down_direct_count,
        Float* __restrict__ surface_down_diffuse_count,
        Float* __restrict__ surface_up_count,
        Float* __restrict__ atmos_direct_count,
        Float* __restrict__ atmos_diffuse_count,
        const Float* __restrict__ k_ext,
        const Optics_scat* __restrict__ scat_asy,
        const Float tod_inc_direct,
        const Float tod_inc_diffuse,
        const Float* __restrict__ surface_albedo,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const int ngrid_x, const int ngrid_y, const int ngrid_z,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot,
        curandDirectionVectors32_t* qrng_vectors, unsigned int* qrng_constants) // const Float* __restrict__ cloud_dims)
{
    const Vector<Float> solar_dir = {dir_x, dir_y, dir_z};
    const Float kgrid_x = x_size/Float(ngrid_x);
    const Float kgrid_y = y_size/Float(ngrid_y);
    const Float kgrid_z = z_size/Float(ngrid_z);

    const int n = blockDim.x * blockIdx.x + threadIdx.x;

    Photon photon;
    Random_number_generator<Float> rng(n);
    Quasi_random_number_generator_2d qrng(qrng_vectors, qrng_constants, n*photons_to_shoot + qrng_gpt_offset*photons_to_shoot*block_size*grid_size);

    const Float s_min = max(z_size, max(y_size, x_size)) * Float_epsilon;

    // Set up the initial photons.
    Int photons_shot = Atomic_reduce_const;
    Float weight;

    reset_photon(
            photon, photons_shot, photons_to_shoot,
            qrng_grid_x, qrng_grid_y,
            toa_down_count,
            qrng, rng,
            tod_inc_direct, tod_inc_diffuse,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            weight, itot, jtot);

    Float tau = Float(0.);
    Float d_max = Float(0.);
    Float k_ext_null;
    Bool transition = false;
    int i_n, j_n, k_n;

    while (photons_shot < photons_to_shoot)
    {
        // if d_max is zero, find current grid and maximum distance
        if (d_max == Float(0.))
        {
            i_n = float_to_int(photon.position.x, kgrid_x, ngrid_x);
            j_n = float_to_int(photon.position.y, kgrid_y, ngrid_y);
            k_n = float_to_int(photon.position.z, kgrid_z, ngrid_z);
            const Float sx = abs((photon.direction.x > 0) ? ((i_n+1) * kgrid_x - photon.position.x)/photon.direction.x : (i_n*kgrid_x - photon.position.x)/photon.direction.x);
            const Float sy = abs((photon.direction.y > 0) ? ((j_n+1) * kgrid_y - photon.position.y)/photon.direction.y : (j_n*kgrid_y - photon.position.y)/photon.direction.y);
            const Float sz = abs((photon.direction.z > 0) ? ((k_n+1) * kgrid_z - photon.position.z)/photon.direction.z : (k_n*kgrid_z - photon.position.z)/photon.direction.z);
            d_max = min(sx, min(sy, sz));
            const int ijk_n = i_n + j_n*ngrid_x + k_n*ngrid_x*ngrid_y;
            k_ext_null = k_null_grid[ijk_n];
        }

        if (!transition)
        {
            tau = sample_tau(rng());
        }
        transition = false;
        const Float dn = max(Float_epsilon, tau / k_ext_null);

        if (dn >= d_max)
        {
            const Float dx = photon.direction.x * (s_min + d_max);
            const Float dy = photon.direction.y * (s_min + d_max);
            const Float dz = photon.direction.z * (s_min + d_max);

            photon.position.x += dx;
            photon.position.y += dy;
            photon.position.z += dz;

            // surface hit
            if (photon.position.z < Float_epsilon)
            {
                photon.position.z = Float_epsilon;
                const int i = float_to_int(photon.position.x, dx_grid, itot);
                const int j = float_to_int(photon.position.y, dy_grid, jtot);
                const int ij = i + j*itot;
                d_max = Float(0.);

                #ifndef NDEBUG
                if (ij < 0 || ij >=itot*jtot) printf("outofbounds 1");
                #endif

                // // Add surface irradiance
                if (photon.kind == Photon_kind::Direct)
                    write_photon_out(&surface_down_direct_count[ij], weight);
                else if (photon.kind == Photon_kind::Diffuse)
                    write_photon_out(&surface_down_diffuse_count[ij], weight);
                // if (from_solar_cone(solar_dir, photon.direction))
                //     write_photon_out(&surface_down_direct_count[ij], weight);
                // else
                //     write_photon_out(&surface_down_diffuse_count[ij], weight);

                // Update weights and add upward surface flux
                const Float local_albedo = surface_albedo[0];
                weight *= local_albedo;
                write_photon_out(&surface_up_count[ij], weight);

                if (weight < w_thres)
                    weight = (rng() > weight) ? Float(0.) : Float(1.);

                // only with nonzero weight continue ray tracing, else start new ray
                if (weight > Float(0.))
                {
                    const Float mu_surface = sqrt(rng());
                    const Float azimuth_surface = Float(2.*M_PI)*rng();

                    photon.direction.x = mu_surface*sin(azimuth_surface);
                    photon.direction.y = mu_surface*cos(azimuth_surface);
                    photon.direction.z = sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon);
                    photon.kind = Photon_kind::Diffuse;
                }
                else
                {
                    reset_photon(
                            photon, photons_shot, photons_to_shoot,
                            qrng_grid_x, qrng_grid_y,
                            toa_down_count,
                            qrng, rng,
                            tod_inc_direct, tod_inc_diffuse,
                            x_size, y_size, z_size,
                            dx_grid, dy_grid, dz_grid,
                            dir_x, dir_y, dir_z,
                            weight, itot, jtot);
                }
            }

            // TOD exit
            else if (photon.position.z >= z_size)
            {
                d_max = Float(0.);

                const int i = float_to_int(photon.position.x, dx_grid, itot);
                const int j = float_to_int(photon.position.y, dy_grid, jtot);
                const int ij = i + j*itot;

                #ifndef NDEBUG
                if (ij < 0 || ij >=itot*jtot) printf("outofbounds 2");
                #endif

                write_photon_out(&tod_up_count[ij], weight);

                reset_photon(
                        photon, photons_shot, photons_to_shoot,
                        qrng_grid_x, qrng_grid_y,
                        toa_down_count,
                        qrng, rng,
                        tod_inc_direct, tod_inc_diffuse,
                        x_size, y_size, z_size,
                        dx_grid, dy_grid, dz_grid,
                        dir_x, dir_y, dir_z,
                        weight, itot, jtot);

            }
            // regular cell crossing: adjust tau and apply periodic BC
            else
            {
                photon.position.x += photon.direction.x>0 ? s_min : -s_min;
                photon.position.y += photon.direction.y>0 ? s_min : -s_min;
                photon.position.z += photon.direction.z>0 ? s_min : -s_min;

                // Cyclic boundary condition in x.
                photon.position.x = fmod(photon.position.x, x_size);
                if (photon.position.x < Float(0.))
                    photon.position.x += x_size;

                // Cyclic boundary condition in y.
                photon.position.y = fmod(photon.position.y, y_size);
                if (photon.position.y < Float(0.))
                    photon.position.y += y_size;

                tau -= d_max * k_ext_null;
                d_max = Float(0.);
                transition = true;
            }
        }
        else
        {
            Float dx = photon.direction.x * dn;
            Float dy = photon.direction.y * dn;
            Float dz = photon.direction.z * dn;

            photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kgrid_x - s_min) : max(photon.position.x + dx, (i_n) * kgrid_x + s_min);
            photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kgrid_y - s_min) : max(photon.position.y + dy, (j_n) * kgrid_y + s_min);
            photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kgrid_z - s_min) : max(photon.position.z + dz, (k_n) * kgrid_z + s_min);

            // Calculate the 3D index.
            const int i = float_to_int(photon.position.x, dx_grid, itot);
            const int j = float_to_int(photon.position.y, dy_grid, jtot);
            const int k = float_to_int(photon.position.z, dz_grid, ktot);
            const int ijk = i + j*itot + k*itot*jtot;

            // Compute probability not being absorbed and store weighted absorption probability
            const Float k_sca_tot = scat_asy[ijk].k_sca_gas + scat_asy[ijk].k_sca_cld + scat_asy[ijk].k_sca_aer;
            const Float ssa_tot = k_sca_tot / k_ext[ijk];

            const Float f_no_abs = Float(1.) - (Float(1.) - ssa_tot) * (k_ext[ijk]/k_ext_null);

            #ifndef NDEBUG
            if (ijk < 0 || ijk >= itot*jtot*ktot) printf("oufofbounds hr \n");
            #endif

            if (photon.kind == Photon_kind::Direct)
                write_photon_out(&atmos_direct_count[ijk], weight*(1-f_no_abs));
            else
                write_photon_out(&atmos_diffuse_count[ijk], weight*(1-f_no_abs));

            // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
            weight *= f_no_abs;
            if (weight < w_thres)
                weight = (rng() > weight) ? Float(0.) : Float(1.);

            // only with nonzero weight continue ray tracing, else start new ray
            if (weight > Float(0.))
            {
                // Null collision.
                if (rng() >= ssa_tot / (ssa_tot - Float(1.) + k_ext_null / k_ext[ijk]))
                {
                    d_max -= dn;
                }
                // Scattering.
                else
                {
                    d_max = Float(0.);
                    // find scatter type: 0 = gas, 1 = cloud, 2 = aerosol
                    const Float scatter_rng = rng();
                    const int scatter_type = scatter_rng < (scat_asy[ijk].k_sca_aer/k_sca_tot) ? 2 :
                                             scatter_rng < ((scat_asy[ijk].k_sca_aer+scat_asy[ijk].k_sca_cld)/k_sca_tot) ? 1 : 0;
                    Float g;
                    switch (scatter_type)
                    {
                        case 0:
                            g = Float(0.);
                            break;
                        case 1:
                            g = min(Float(1.) - Float_epsilon, scat_asy[ijk].asy_cld);
                            break;
                        case 2:
                            g = min(Float(1.) - Float_epsilon, scat_asy[ijk].asy_aer);
                            break;
                    }

                    const Float cos_scat = scatter_type == 0 ? rayleigh(rng()) : henyey(g, rng());
                    const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                    Vector<Float> t1{Float(0.), Float(0.), Float(0.)};
                    if (fabs(photon.direction.x) < fabs(photon.direction.y))
                    {
                        if (fabs(photon.direction.x) < fabs(photon.direction.z))
                            t1.x = Float(1.);
                        else
                            t1.z = Float(1.);
                    }
                    else
                    {
                        if (fabs(photon.direction.y) < fabs(photon.direction.z))
                            t1.y = Float(1.);
                        else
                            t1.z = Float(1.);
                    }
                    t1 = normalize(t1 - photon.direction*dot(t1, photon.direction));
                    Vector<Float> t2 = cross(photon.direction, t1);

                    const Float phi = Float(2.*M_PI)*rng();

                    photon.direction = cos_scat*photon.direction
                            + sin_scat*(sin(phi)*t1 + cos(phi)*t2);

                    photon.kind = Photon_kind::Diffuse;
                }
            }
            else
            {
                d_max = Float(0.);
                reset_photon(
                        photon, photons_shot, photons_to_shoot,
                        qrng_grid_x, qrng_grid_y,
                        toa_down_count,
                        qrng, rng,
                        tod_inc_direct, tod_inc_diffuse,
                        x_size, y_size, z_size,
                        dx_grid, dy_grid, dz_grid,
                        dir_x, dir_y, dir_z,
                        weight, itot, jtot);

            }
        }
    }
}
