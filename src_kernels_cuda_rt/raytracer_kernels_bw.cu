#include <float.h>
#include <curand_kernel.h>
#include "raytracer_kernels_bw.h"
#include <iomanip>
#include <iostream>

namespace
{
    // using Int = unsigned long long;
    const Int Atomic_reduce_const = (Int)(-1LL);

    //using Int = unsigned int;
    //const Int Atomic_reduce_const = (Int)(-1);

    #ifdef RTE_RRTMGP_SINGLE_PRECISION
    // using Float = float;
    const Float Float_epsilon = FLT_EPSILON;
    // constexpr int block_size = 512;
    // constexpr int grid_size = 64;
    #else
    // using Float = double;
    const Float Float_epsilon = DBL_EPSILON;
    // constexpr int block_size = 512;
    // constexpr int grid_size = 64;
    #endif

    struct Vector
    {
        Float x;
        Float y;
        Float z;

    };

    constexpr Float w_thres = 0.5;
    constexpr Float fov = 160./180.*M_PI;
    // constexpr Float fov_v = 60./180.*M_PI;

    // angle w.r.t. vertical: 0 degrees is looking up, 180 degrees down
    constexpr Float zenith_cam = 0./180.*M_PI; //60./180.*M_PI;
    // angle w.r.t. north,: 0 degrees is looking north
    //constexpr Float azimuth_cam = 0./180*M_PI;
    constexpr Vector upward_cam = {0, 1, 0};

    constexpr Float half_angle = .26656288/180. * M_PI; // sun has a half angle of .266 degrees
    //constexpr Float cos_half_angle = Float(0.9961946980917455); //Float(0.9999891776066407); // cos(half_angle);
    constexpr Float cos_half_angle = Float(0.9999891776066407); // cos(half_angle);
    constexpr Float cos_half_angle_app = cos_half_angle; //we set this a bit larger that cos_half_angle to make sun larger
    constexpr Float solid_angle = Float(6.799910294339209e-05); // 2.*M_PI*(1-cos_half_angle);
    //constexpr Float solid_angle = Float(0.023909417039326832); //Float(6.799910294339209e-05); // 2.*M_PI*(1-cos_half_angle);



    static inline __device__
    Vector operator*(const Vector v, const Float s) { return Vector{s*v.x, s*v.y, s*v.z}; }
    static inline __device__
    Vector operator*(const Float s, const Vector v) { return Vector{s*v.x, s*v.y, s*v.z}; }
    static inline __device__
    Vector operator-(const Vector v1, const Vector v2) { return Vector{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
    static inline __device__
    Vector operator+(const Vector v1, const Vector v2) { return Vector{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }
    static inline __device__
    Vector operator/(const Vector v, const Float s) { return Vector{v.x/s, v.y/s, v.z/s}; }
    static inline __device__
    Vector operator/(const Float s, const Vector v) { return Vector{v.x/s, v.y/s, v.z/s}; }

    __device__
    Vector cross(const Vector v1, const Vector v2)
    {
        return Vector{
                v1.y*v2.z - v1.z*v2.y,
                v1.z*v2.x - v1.x*v2.z,
                v1.x*v2.y - v1.y*v2.x};
    }

    __device__
    Float dot(const Vector& v1, const Vector& v2)
    {
        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
    }

    __device__
    Float norm(const Vector v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }


    __device__
    Vector normalize(const Vector v)
    {
        const Float length = norm(v);
        return Vector{ v.x/length, v.y/length, v.z/length};
    }

    enum class Photon_kind { Direct, Diffuse };

    enum class Phase_kind {Lambertian, Specular, Rayleigh, HG};

    struct Photon
    {
        Vector position;
        Vector direction;
        Photon_kind kind;
        Vector start;
    };


    __device__
    Float pow2(const Float d) { return d*d; }

    __device__
    Vector specular(const Vector dir_in)
    {
        return Vector{dir_in.x, dir_in.y, Float(-1)*dir_in.z};
    }

    __device__
    Float rayleigh(const Float random_number)
    {
        const Float q = Float(4.)*random_number - Float(2.);
        const Float d = Float(1.) + pow2(q);
        const Float u = pow(-q + sqrt(d), Float(1./3.));
        return u - Float(1.)/u;
    }


    __device__
    Float henyey(const Float g, const Float random_number)
    {
        const Float a = pow2(Float(1.) - pow2(g));
        const Float b = Float(2.)*g*pow2(Float(2.)*random_number*g + Float(1.) - g);
        const Float c = -g/Float(2.) - Float(1.)/(Float(2.)*g);
        return Float(-1.)*(a/b) - c;
    }

    __device__
    Float lambertian_phase()
    {
        return Float(1.)/M_PI;
    }

    __device__
    Float rayleigh_phase(const Float cos_angle)
    {
        return Float(3.)/(Float(16.)*M_PI) * (1+cos_angle*cos_angle);
    }

    __device__
    Float henyey_phase(const Float g, const Float cos_angle)
    {
        const Float denom = 1 + g*g - 2*g*cos_angle;
        return Float(1.)/(Float(4.)*M_PI) * (1-g*g) / (denom*sqrt(denom));
    }

    __device__
    Float sample_tau(const Float random_number)
    {
        // Prevent log(0) possibility.
        return Float(-1.)*log(-random_number + Float(1.) + Float_epsilon);
    }

    __device__
    inline int float_to_int(const Float s_size, const Float ds, const int ntot_max)
    {
        const int ntot = static_cast<int>(s_size / ds);
        return ntot < ntot_max ? ntot : ntot_max-1;
    }

    template<typename T>
    struct Random_number_generator
    {
        __device__ Random_number_generator(unsigned int tid)
        {
            curand_init(tid, tid, 0, &state);
        }

        __device__ T operator()();

        curandState state;
    };


    template<>
    __device__ double Random_number_generator<double>::operator()()
    {
        return 1. - curand_uniform_double(&state);
    }


    template<>
    __device__ float Random_number_generator<float>::operator()()
    {
        return 1.f - curand_uniform(&state);
    }

    __device__
    Float transmission_direct_sun(
            Photon photon,
            const int n,
            Random_number_generator<Float>& rng,
            const Vector& sun_dir,
            const Grid_knull* __restrict__ k_null_grid,
            const Optics_ext* __restrict__ k_ext,
            const Float* __restrict__ bg_tau_cum,
            const Float* __restrict__ z_lev_bg,
            const int bg_idx,
            const Float kgrid_x, const Float kgrid_y, const Float kgrid_z,
            const Float dx_grid, const Float dy_grid, const Float dz_grid,
            const Float x_size, const Float y_size, const Float z_size,
            const int itot, const int jtot, const int ktot,
            const Float s_min, const Float s_min_bg)

    {
        Float tau;
        Float k_ext_null;
        Float k_ext_min;
        Float d_max = Float(0.);
        Float tau_min = Float(0.);

        bool transition = false;
        int i_n,j_n,k_n;
        while (true)
        {
            if (!transition)
            {
                tau = sample_tau(rng());
            }
            transition = false;

            if (photon.position.z > z_size)
            {
                //printf("x %f \n",tau_min + bg_tau_cum[bg_idx]);
                return exp(Float(-1.) * (tau_min + bg_tau_cum[bg_idx]));
            }
            // main grid (dynamical)
            else
            {
                // distance to nearest boundary of acceleration gid voxel
                if (d_max == Float(0.))
                {
                    i_n = float_to_int(photon.position.x, kgrid_x, ngrid_x);
                    j_n = float_to_int(photon.position.y, kgrid_y, ngrid_y);
                    k_n = float_to_int(photon.position.z, kgrid_z, ngrid_z);
                    const Float sx = abs((sun_dir.x > 0) ? ((i_n+1) * kgrid_x - photon.position.x)/sun_dir.x : (i_n*kgrid_x - photon.position.x)/sun_dir.x);
                    const Float sy = abs((sun_dir.y > 0) ? ((j_n+1) * kgrid_y - photon.position.y)/sun_dir.y : (j_n*kgrid_y - photon.position.y)/sun_dir.y);
                    const Float sz = ((k_n+1) * kgrid_z - photon.position.z)/sun_dir.z;
                    d_max = min(sx, min(sy, sz));
                    const int ijk = i_n + j_n*ngrid_x + k_n*ngrid_x*ngrid_y;

                    // decomposition tracking: minimum k_ext is used to integrate transmissivity, difference max-min as k_null
                    k_ext_min  = k_null_grid[ijk].k_min;
                    k_ext_null = k_null_grid[ijk].k_max - k_ext_min;
                }

                const Float dn = max(Float_epsilon, tau / k_ext_null);
                if (dn >= d_max)
                {
                    // update position
                    tau_min += k_ext_min * d_max;
                    const Float dx = sun_dir.x * d_max;
                    const Float dy = sun_dir.y * d_max;
                    const Float dz = sun_dir.z * d_max;

                    photon.position.x += dx;
                    photon.position.y += dy;
                    photon.position.z += dz;

                    // TOA exit
                    if (photon.position.z >= z_size - s_min)
                    {
                        photon.position.z = z_size + s_min_bg;
                        //return exp(-tau_min);
                    }
                    // regular cell crossing: adjust tau and apply periodic BC
                    else
                    {
                        photon.position.x += sun_dir.x>0 ? s_min : -s_min;
                        photon.position.y += sun_dir.y>0 ? s_min : -s_min;
                        photon.position.z += sun_dir.z>0 ? s_min : -s_min;

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
                    // hit event: update event and evaluuate change the hit is a null collision
                    tau_min += k_ext_min * dn;
                    const Float dx = sun_dir.x * dn;
                    const Float dy = sun_dir.y * dn;
                    const Float dz = sun_dir.z * dn;

                    photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kgrid_x - s_min) : max(photon.position.x + dx, (i_n) * kgrid_x + s_min);
                    photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kgrid_y - s_min) : max(photon.position.y + dy, (j_n) * kgrid_y + s_min);
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kgrid_z - s_min) : max(photon.position.z + dz, (k_n) * kgrid_z + s_min);

                    // Calculate the 3D index.
                    const int i = float_to_int(photon.position.x, dx_grid, itot);
                    const int j = float_to_int(photon.position.y, dy_grid, jtot);
                    const int k = float_to_int(photon.position.z, dz_grid, ktot);
                    const int ijk = i + j*itot + k*itot*jtot;

                    // Handle the action.
                    const Float k_ext_tot = k_ext[ijk].gas + k_ext[ijk].cloud - k_ext_min;
                    // Compute probability not being absorbed and store weighted absorption probability
                    if (rng() < k_ext_tot/k_ext_null) return 0;

                    d_max -= dn;

                }
            }
        }
    }

    __device__
    inline void reset_photon(
            Photon& photon,
            Float* __restrict__ camera_count,
            Float* __restrict__ camera_shot,
            Int& photons_shot,
            const int ij_cam, const int n,
            Random_number_generator<Float>& rng,
            const Vector& sun_direction,
            const Grid_knull* __restrict__ k_null_grid,
            const Optics_ext* __restrict__ k_ext,
            const Float kgrid_x, const Float kgrid_y, const Float kgrid_z,
            const Float x_size, const Float y_size, const Float z_size,
            const Float dx_grid, const Float dy_grid, const Float dz_grid,
            const Float dir_x, const Float dir_y, const Float dir_z,
            const bool generation_completed, Float& weight, int& bg_idx,
            const int cam_nx, const int cam_ny, const Float* __restrict__ cam_data,
            const Vector axis_h, const Vector axis_v, const Vector axis_z,
            const int itot, const int jtot, const int ktot, const int kbg,
            const Float* __restrict__ bg_tau_cum,
            const Float* __restrict__ z_lev_bg,
            const Float s_min,
            const Float s_min_bg)
    {
        ++photons_shot;
        if (!generation_completed)
        {
            //const Float i = (ij_cam % cam_nx) / Float(cam_nx) + Float(0.5) / Float(cam_nx);//(Float(0.5) * Float(cam_nx)) - Float(1.) + Float(0.5) / Float(cam_nx);
            //const Float j = Float(ij_cam/cam_nx) / (Float(0.5) * Float(cam_nx)) - Float(1.) + Float(0.5) / Float(cam_nx);
            const Float i = (Float(ij_cam % cam_nx) + rng())/ Float(cam_nx);//(Float(0.5) * Float(cam_nx)) - Float(1.) + Float(0.5) / Float(cam_nx);
            const Float j = (Float(ij_cam / cam_nx) + rng())/ Float(cam_nx);//(Float(0.5) * Float(cam_nx)) - Float(1.) + Float(0.5) / Float(cam_nx);


            photon.position.x = Float(3225.) + s_min;
            photon.position.y = Float(3225.) + s_min;
            photon.position.z = cam_data[1] + s_min; //Float(500.)+ s_min;

            const Float photon_zenith = i * Float(0.5) * M_PI;
            const Float photon_azimuth = j * Float(2.) * M_PI;

            photon.direction = {sin(photon_zenith) * sin(photon_azimuth), sin(photon_zenith) * cos(photon_azimuth), cos(photon_zenith)};

            photon.kind = Photon_kind::Direct;
            weight = 1;
            bg_idx = 0;

            if ( (dot(photon.direction, sun_direction) > cos_half_angle_app) )
            {
                const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                           k_null_grid,k_ext,
                                           bg_tau_cum, z_lev_bg, bg_idx,
                                           kgrid_x, kgrid_y, kgrid_z,
                                           dx_grid, dy_grid, dz_grid,
                                           x_size, y_size, z_size,
                                           itot, jtot, ktot,
                                           s_min, s_min_bg);

               atomicAdd(&camera_count[ij_cam], weight * trans_sun);
            }
            atomicAdd(&camera_shot[ij_cam], Float(1.));
        }
    }

    __device__
    inline Float probability_from_sun(
            Photon photon, Vector sun_direction, const Float solid_angle, const Float g, const Phase_kind kind)
    {
        const Float cos_angle = dot(photon.direction, sun_direction);
        if (kind == Phase_kind::HG)
        {
            return henyey_phase(g, cos_angle) * solid_angle;
        }
        else if (kind == Phase_kind::Rayleigh)
        {
            return rayleigh_phase(cos_angle) * solid_angle;
        }
        else if (kind == Phase_kind::Lambertian)
        {
            return lambertian_phase() * solid_angle;
        }
        else if (kind == Phase_kind::Specular)
        {
            return (dot( specular(photon.direction) , sun_direction) > cos_half_angle_app) ? Float(1.) : Float(0.);
        }
    }

    __device__
    inline void write_photon_out(Float* field_out, const Float w)
    {
        atomicAdd(field_out, w);
    }

}


__global__
void ray_tracer_kernel_bw(
        const Int photons_to_shoot,
        const Grid_knull* __restrict__ k_null_grid,
        Float* __restrict__ camera_count,
        Float* __restrict__ camera_shot,
        int* __restrict__ counter,
        const int cam_nx, const int cam_ny, const Float* __restrict__ cam_data,
        const Optics_ext* __restrict__ k_ext, const Optics_scat* __restrict__ ssa_asy,
        const Optics_ext* __restrict__ k_ext_bg, const Optics_scat* __restrict__ ssa_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ surface_albedo,
        const Float mu,
        const Float x_size, const Float y_size, const Float z_size,
        const Float dx_grid, const Float dy_grid, const Float dz_grid,
        const Float dir_x, const Float dir_y, const Float dir_z,
        const int itot, const int jtot, const int ktot, const int kbg)
{
    const Phase_kind surface_kind = Phase_kind::Lambertian;
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    Vector sun_direction = {-dir_x, -dir_y, -dir_z};
    Vector normal_direction = {0, 0, 1};

    const Float azimuth_cam = cam_data[0];
    const Vector cam = {sin(zenith_cam) * sin(azimuth_cam), sin(zenith_cam) * cos(azimuth_cam), cos(zenith_cam) };
    const Vector axis_h = normalize(cross(cam, upward_cam));
    const Vector axis_v = normalize(cross(cam, axis_h));
    const Vector axis_z = cam / tan(fov/Float(2.));

    extern __shared__ Float bg_tau_cum[];
    Float bg_tau = Float(0.);
    for (int k=kbg-1; k >= 0; --k)
    {
    //    if (n==0)printf("X %e %e %f \n",k_ext_bg[k].gas , k_ext_bg[k].cloud,  abs(z_lev_bg[k+1]-z_lev_bg[k]));
        bg_tau += (k_ext_bg[k].gas + k_ext_bg[k].cloud) * abs(z_lev_bg[k+1]-z_lev_bg[k]) / mu;
        bg_tau_cum[k] = bg_tau;
    }
    const Float bg_transmissivity = exp(-bg_tau_cum[0]);
    //if (bg_transmissivity < Float(1.e-4)) return;

    const Float kgrid_x = x_size/Float(ngrid_x);
    const Float kgrid_y = y_size/Float(ngrid_y);
    const Float kgrid_z = z_size/Float(ngrid_z);
    const Float z_top = z_lev_bg[kbg];

    Random_number_generator<Float> rng(n);

    const Float s_min = max(max(z_size, x_size), y_size) * Float_epsilon;
    const Float s_min_bg = max(max(x_size, y_size), z_top) * Float_epsilon;

    const int pixels_per_thread = cam_nx * cam_ny / (grid_size * block_size);
    const int photons_per_pixel = photons_to_shoot / pixels_per_thread ;

    while (counter[0] < cam_nx*cam_ny)
    {
        const int ij_cam = atomicAdd(&counter[0], 1);

        if (ij_cam >= cam_nx*cam_ny)
            return;

        const int i = ij_cam % cam_nx;
        const int j = ij_cam / cam_nx;

        const bool completed = false;
        Int photons_shot = Atomic_reduce_const;
        Float weight;
        int bg_idx;

        Photon photon;
        reset_photon(
                photon, camera_count,camera_shot, photons_shot,
                ij_cam, n, rng, sun_direction,
                k_null_grid, k_ext,
                kgrid_x, kgrid_y, kgrid_z,
                x_size, y_size, z_size,
                dx_grid, dy_grid, dz_grid,
                dir_x, dir_y, dir_z,
                completed, weight, bg_idx,
                cam_nx, cam_ny, cam_data, axis_h, axis_v, axis_z,
                itot, jtot, ktot, kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);

        Float tau;
        Float d_max = Float(0.);
        Float k_ext_null;
        bool transition = false;
        int i_n, j_n, k_n, ijk_n;
        bool m = true;

        while (photons_shot < photons_per_pixel)
        {
            const bool photon_generation_completed = (photons_shot == photons_per_pixel - 1);

            if (!transition)
            {
                tau = sample_tau(rng());
            }
            transition = false;

            // 1D raytracing between TOD and TOA?
            if (photon.position.z > z_size)
            {
                const Float k_ext_bg_tot = k_ext_bg[bg_idx].gas + k_ext_bg[bg_idx].cloud;
                const Float dn = max(Float_epsilon, tau / k_ext_bg_tot);
                d_max = abs( (photon.direction.z>0) ? (z_lev_bg[bg_idx+1] - photon.position.z) / photon.direction.z : (z_lev_bg[bg_idx] - photon.position.z) / photon.direction.z );
                if (dn >= d_max)
                {
                    photon.position.z = (photon.direction.z > 0) ? z_lev_bg[bg_idx+1] + s_min_bg : z_lev_bg[bg_idx] - s_min_bg;
                    photon.position.y += photon.direction.y * d_max;
                    photon.position.x += photon.direction.x * d_max;

                    // move to actual grid: reduce tau and set next position
                    if (photon.position.z <= z_size + s_min_bg)
                    {
                        tau -= k_ext_bg_tot * (d_max + s_min_bg);
                        photon.position.z = z_size - s_min;
                        d_max = Float(0.);
                        transition=true;

                        // Cyclic boundary condition in x.
                        photon.position.x = fmod(photon.position.x, x_size);
                        if (photon.position.x < Float(0.))
                            photon.position.x += x_size;

                        // Cyclic boundary condition in y.
                        photon.position.y = fmod(photon.position.y, y_size);
                        if (photon.position.y < Float(0.))
                            photon.position.y += y_size;
                    }
                    else if (photon.position.z >= z_top)
                    {
                        // Leaving top-of-domain
                        d_max = Float(0.);
                        reset_photon(
                                photon, camera_count,camera_shot, photons_shot,
                                ij_cam, n, rng, sun_direction,
                                k_null_grid, k_ext,
                                kgrid_x, kgrid_y, kgrid_z,
                                x_size, y_size, z_size,
                                dx_grid, dy_grid, dz_grid,
                                dir_x, dir_y, dir_z,
                                completed, weight, bg_idx,
                                cam_nx, cam_ny, cam_data, axis_h, axis_v, axis_z,
                                itot, jtot, ktot, kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                    }
                    else
                    {
                        // just move to next grid
                        transition = true;
                        tau -= k_ext_bg_tot * (d_max + s_min_bg);

                        bg_idx += (photon.direction.z > 0) ? 1 : -1;
                    }
                }
                else
                {
                    const Float dz = photon.direction.z * dn;
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, z_lev_bg[bg_idx+1] - s_min_bg) : max(photon.position.z + dz, z_lev_bg[bg_idx] + s_min_bg);

                    photon.position.y += photon.direction.y * dn;
                    photon.position.x += photon.direction.x * dn;

                    // Compute probability not being absorbed and store weighted absorption probability
                    const Float f_no_abs = ssa_asy_bg[bg_idx].ssa;

                    // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
                    weight *= f_no_abs;
                    if (weight < w_thres)
                        weight = (rng() > weight) ? Float(0.) : Float(1.);

                    // only with nonzero weight continue ray tracing, else start new ray
                    if (weight > Float(0.))
                    {
                        // SUN SCATTERING GOES HERE
                        const bool cloud_scatter = rng() < (k_ext_bg[bg_idx].cloud / k_ext_bg_tot);
                        const Float g = cloud_scatter ? ssa_asy_bg[bg_idx].asy : Float(0.);
                        const Float cos_scat = cloud_scatter ? henyey(g, rng()) : rayleigh(rng());
                        const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                        // direct contribution
                        const Phase_kind kind = cloud_scatter ? Phase_kind::HG :Phase_kind::Rayleigh;
                        const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, g, kind);
                        const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                    k_null_grid,k_ext,
                                                    bg_tau_cum, z_lev_bg, bg_idx,
                                                    kgrid_x, kgrid_y, kgrid_z,
                                                    dx_grid, dy_grid, dz_grid,
                                                    x_size, y_size, z_size,
                                                    itot, jtot, ktot,
                                                    s_min, s_min_bg);
                        atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);


                        Vector t1{Float(0.), Float(0.), Float(0.)};
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
                        Vector t2 = cross(photon.direction, t1);

                        const Float phi = Float(2.*M_PI)*rng();

                        photon.direction = cos_scat*photon.direction
                                + sin_scat*(sin(phi)*t1 + cos(phi)*t2);

                        photon.kind = Photon_kind::Diffuse;
                    }
                    else
                    {
                        d_max = Float(0.);
                        reset_photon(
                                photon, camera_count,camera_shot, photons_shot,
                                ij_cam, n, rng, sun_direction,
                                k_null_grid, k_ext,
                                kgrid_x, kgrid_y, kgrid_z,
                                x_size, y_size, z_size,
                                dx_grid, dy_grid, dz_grid,
                                dir_x, dir_y, dir_z,
                                photon_generation_completed, weight, bg_idx,
                                cam_nx, cam_ny, cam_data, axis_h, axis_v, axis_z,
                                itot, jtot, ktot, kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                    }
                }
            }
            // we reached the 'dynamical' domain, now things get interesting
            else
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
                    ijk_n = i_n + j_n*ngrid_x + k_n*ngrid_y*ngrid_x;
                    k_ext_null = k_null_grid[ijk_n].k_max;
                }

                const Float dn = max(Float_epsilon, tau / k_ext_null);

                if ( ( dn >= d_max) )
                {
                    const Float dx = photon.direction.x * (d_max);
                    const Float dy = photon.direction.y * (d_max);
                    const Float dz = photon.direction.z * (d_max);

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

                        // Update weights and add upward surface flux
                        const Float local_albedo = surface_albedo[0];
                        weight *= local_albedo;

                        // SUN SCATTERING GOES HERE
                        const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, Float(0.), (photon.kind == Photon_kind::Direct) ? surface_kind : Phase_kind::Lambertian);
                        const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                    k_null_grid,k_ext,
                                                    bg_tau_cum, z_lev_bg, bg_idx,
                                                    kgrid_x, kgrid_y, kgrid_z,
                                                    dx_grid, dy_grid, dz_grid,
                                                    x_size, y_size, z_size,
                                                    itot, jtot, ktot,
                                                    s_min, s_min_bg);
                        atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);

                        if (weight < w_thres)
                            weight = (rng() > weight) ? Float(0.) : Float(1.);

                        // only with nonzero weight continue ray tracing, else start new ray
                        if (weight > Float(0.))
                        {
                            if (surface_kind == Phase_kind::Lambertian)
                            {
                                const Float mu_surface = sqrt(rng());
                                const Float azimuth_surface = Float(2.*M_PI)*rng();

                                photon.direction.x = mu_surface*sin(azimuth_surface);
                                photon.direction.y = mu_surface*cos(azimuth_surface);
                                photon.direction.z = sqrt(Float(1.) - mu_surface*mu_surface + Float_epsilon);
                                photon.kind = Photon_kind::Diffuse;
                            }
                            else if (surface_kind == Phase_kind::Specular)
                            {
                                photon.direction = specular(photon.direction);
                            }
                        }
                        else
                        {
                            reset_photon(
                                    photon, camera_count,camera_shot, photons_shot,
                                    ij_cam, n, rng, sun_direction,
                                    k_null_grid, k_ext,
                                    kgrid_x, kgrid_y, kgrid_z,
                                    x_size, y_size, z_size,
                                    dx_grid, dy_grid, dz_grid,
                                    dir_x, dir_y, dir_z,
                                    photon_generation_completed, weight, bg_idx,
                                    cam_nx, cam_ny, cam_data, axis_h, axis_v, axis_z,
                                    itot, jtot, ktot, kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                        }
                    }

                    // TOD exit
                    else if (photon.position.z >= z_size)
                    {
                        photon.position.z = z_size + s_min_bg;
                        tau -= d_max * k_ext_null;
                        bg_idx = 0;
                        d_max = Float(0.);
                        transition = true;

                        // const Float p_sun = probability_from_sun(photon, normal_direction, solid_angle, Float(0.), Phase_kind::Lambertian);
                        // //atomicAdd(&camera_diff[ij_cam], weight * p_sun * bg_trans);
                        // //atomicAdd(&camera_count[ij_cam], weight * bg_trans * tod_frac_diff);

                        // d_max = Float(0.);
                        // reset_photon(
                        //         photon, camera_count,camera_shot, photons_shot,
                        //         ij_cam, n, rng, sun_direction,
                        //         k_null_grid, k_ext,
                        //         kgrid_x, kgrid_y, kgrid_z,
                        //         x_size, y_size, z_size,
                        //         dx_grid, dy_grid, dz_grid,
                        //         dir_x, dir_y, dir_z,
                        //         photon_generation_completed, weight,
                        //         cam_nx, cam_ny, cam_data, axis_h, axis_v, axis_z,
                        //         itot, jtot, ktot, bg_trans, s_min);
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
                    const Float dx = photon.direction.x * dn;
                    const Float dy = photon.direction.y * dn;
                    const Float dz = photon.direction.z * dn;

                    photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kgrid_x - s_min) : max(photon.position.x + dx, (i_n) * kgrid_x + s_min);
                    photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kgrid_y - s_min) : max(photon.position.y + dy, (j_n) * kgrid_y + s_min);
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kgrid_z - s_min) : max(photon.position.z + dz, (k_n) * kgrid_z + s_min);

                    // Calculate the 3D index.
                    const int i = float_to_int(photon.position.x, dx_grid, itot);
                    const int j = float_to_int(photon.position.y, dy_grid, jtot);
                    const int k = float_to_int(photon.position.z, dz_grid, ktot);
                    const int ijk = i + j*itot + k*itot*jtot;

                    // Handle the action.
                    const Float random_number = rng();
                    const Float k_ext_tot = k_ext[ijk].gas + k_ext[ijk].cloud;

                    // Compute probability not being absorbed and store weighted absorption probability
                    const Float f_no_abs = Float(1.) - (Float(1.) - ssa_asy[ijk].ssa) * (k_ext_tot/k_ext_null);

                    // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
                    weight *= f_no_abs;

                    if (weight < w_thres)
                        weight = (rng() > weight) ? Float(0.) : Float(1.);

                    // only with nonzero weight continue ray tracing, else start new ray
                    if (weight > Float(0.))
                    {
                        // Null collision.
                        if (random_number >= ssa_asy[ijk].ssa / (ssa_asy[ijk].ssa - Float(1.) + k_ext_null / k_ext_tot))
                        {
                            d_max -= dn;
                        }
                        // Scattering.
                        else
                        {
                            d_max = Float(0.);
                            const bool cloud_scatter = rng() < (k_ext[ijk].cloud / k_ext_tot);
                            const Float g = cloud_scatter ? ssa_asy[ijk].asy : Float(0.);
                            const Float cos_scat = cloud_scatter ? henyey(g, rng()) : rayleigh(rng());
                            const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                            // SUN SCATTERING GOES HERE
                            const Phase_kind kind = cloud_scatter ? Phase_kind::HG : Phase_kind::Rayleigh;
                            const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, g, kind);
                            const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                        k_null_grid,k_ext,
                                                        bg_tau_cum, z_lev_bg, bg_idx,
                                                        kgrid_x, kgrid_y, kgrid_z,
                                                        dx_grid, dy_grid, dz_grid,
                                                        x_size, y_size, z_size,
                                                        itot, jtot, ktot,
                                                        s_min, s_min_bg);
                            atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);

                            Vector t1{Float(0.), Float(0.), Float(0.)};
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
                            Vector t2 = cross(photon.direction, t1);

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
                                photon, camera_count,camera_shot, photons_shot,
                                ij_cam, n, rng, sun_direction,
                                k_null_grid, k_ext,
                                kgrid_x, kgrid_y, kgrid_z,
                                x_size, y_size, z_size,
                                dx_grid, dy_grid, dz_grid,
                                dir_x, dir_y, dir_z,
                                photon_generation_completed, weight, bg_idx,
                                cam_nx, cam_ny, cam_data, axis_h, axis_v, axis_z,
                                itot, jtot, ktot, kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                    }
                }
            }
        }
    }
}

