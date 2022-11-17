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

    constexpr Float w_thres = 0.5;
    constexpr Float fov = 160./180.*M_PI;
    // constexpr Float fov_v = 60./180.*M_PI;

    // angle w.r.t. vertical: 0 degrees is looking up, 180 degrees down
    constexpr Float zenith_cam = 0./180.*M_PI; //60./180.*M_PI;
    // angle w.r.t. north,: 0 degrees is looking north
    //constexpr Float azimuth_cam = 0./180*M_PI;
    constexpr Vector<Float> upward_cam = {0, 1, 0};

    constexpr Float half_angle = .26656288/180. * M_PI; // sun has a half angle of .266 degrees
    //constexpr Float cos_half_angle = Float(0.9961946980917455); //Float(0.9999891776066407); // cos(half_angle);
    constexpr Float cos_half_angle = Float(0.9999891776066407); // cos(half_angle);
    constexpr Float cos_half_angle_app = cos_half_angle; //we set this a bit larger that cos_half_angle to make sun larger
    constexpr Float solid_angle = Float(6.799910294339209e-05); // 2.*M_PI*(1-cos_half_angle);
    //constexpr Float solid_angle = Float(0.023909417039326832); //Float(6.799910294339209e-05); // 2.*M_PI*(1-cos_half_angle);

    template<typename T> __device__
    Vector<T> cross(const Vector<T> v1, const Vector<T> v2)
    {
        return Vector<T>{
                v1.y*v2.z - v1.z*v2.y,
                v1.z*v2.x - v1.x*v2.z,
                v1.x*v2.y - v1.y*v2.x};
    }

    template<typename T> __device__
    Float dot(const Vector<T>& v1, const Vector<T>& v2)
    {
        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
    }

    template<typename T> __device__
    Float norm(const Vector<T> v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }


    template<typename T> __device__
    Vector<T> normalize(const Vector<T> v)
    {
        const Float length = norm(v);
        return Vector<T>{ v.x/length, v.y/length, v.z/length};
    }

    enum class Photon_kind { Direct, Diffuse };

    enum class Phase_kind {Lambertian, Specular, Rayleigh, HG};

    struct Photon
    {
        Vector<Float> position;
        Vector<Float> direction;
        Photon_kind kind;
        Vector<Float> start;
    };


    __device__
    Float pow2(const Float d) { return d*d; }

    template<typename T> __device__
    Vector<T> specular(const Vector<T> dir_in)
    {
        return Vector<T>{dir_in.x, dir_in.y, Float(-1)*dir_in.z};
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
        const Float denom = max(Float_epsilon, 1 + g*g - 2*g*cos_angle);
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
            const Vector<Float>& sun_dir,
            const Grid_knull* __restrict__ k_null_grid,
            const Float* __restrict__ k_ext,
            const Float* __restrict__ bg_tau_cum,
            const Float* __restrict__ z_lev_bg,
            const int bg_idx,
            const Vector<int>& kn_grid,
            const Vector<Float>& kn_grid_d,
            const Vector<Float>& grid_d,
            const Vector<Float>& grid_size,
            const Vector<int>& grid_cells,
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

            if (photon.position.z > grid_size.z)
            {
                return exp(Float(-1.) * (tau_min + bg_tau_cum[bg_idx]));
            }
            // main grid (dynamical)
            else
            {
                // distance to nearest boundary of acceleration gid voxel
                if (d_max == Float(0.))
                {
                    i_n = float_to_int(photon.position.x, kn_grid_d.x, kn_grid.x);
                    j_n = float_to_int(photon.position.y, kn_grid_d.y, kn_grid.y);
                    k_n = float_to_int(photon.position.z, kn_grid_d.z, kn_grid.z);
                    const Float sx = abs((sun_dir.x > 0) ? ((i_n+1) * kn_grid_d.x - photon.position.x)/sun_dir.x : (i_n*kn_grid_d.x - photon.position.x)/sun_dir.x);
                    const Float sy = abs((sun_dir.y > 0) ? ((j_n+1) * kn_grid_d.y - photon.position.y)/sun_dir.y : (j_n*kn_grid_d.y - photon.position.y)/sun_dir.y);
                    const Float sz = ((k_n+1) * kn_grid_d.z - photon.position.z)/sun_dir.z;
                    d_max = min(sx, min(sy, sz));
                    const int ijk = i_n + j_n*kn_grid.x + k_n*kn_grid.x*kn_grid.y;

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
                    if (photon.position.z >= grid_size.z - s_min)
                    {
                        photon.position.z = grid_size.z + s_min_bg;
                        //return exp(-tau_min);
                    }
                    // regular cell crossing: adjust tau and apply periodic BC
                    else
                    {
                        photon.position.x += sun_dir.x>0 ? s_min : -s_min;
                        photon.position.y += sun_dir.y>0 ? s_min : -s_min;
                        photon.position.z += sun_dir.z>0 ? s_min : -s_min;

                        // Cyclic boundary condition in x.
                        photon.position.x = fmod(photon.position.x, grid_size.x);
                        if (photon.position.x < Float(0.))
                            photon.position.x += grid_size.x;

                        // Cyclic boundary condition in y.
                        photon.position.y = fmod(photon.position.y, grid_size.y);
                        if (photon.position.y < Float(0.))
                            photon.position.y += grid_size.y;

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

                    photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kn_grid_d.x - s_min) : max(photon.position.x + dx, (i_n) * kn_grid_d.x + s_min);
                    photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kn_grid_d.y - s_min) : max(photon.position.y + dy, (j_n) * kn_grid_d.y + s_min);
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kn_grid_d.z - s_min) : max(photon.position.z + dz, (k_n) * kn_grid_d.z + s_min);

                    // Calculate the 3D index.
                    const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                    const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                    const int k = float_to_int(photon.position.z, grid_d.z, grid_cells.z);
                    const int ijk = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;

                    // Handle the action.
                    const Float k_ext_tot = k_ext[ijk] - k_ext_min;
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
            const Vector<Float>& sun_direction,
            const Grid_knull* __restrict__ k_null_grid,
            const Float* __restrict__ k_ext,
            const Vector<int>& kn_grid,
            const Vector<Float>& kn_grid_d,
            const Vector<Float>& grid_d,
            const Vector<Float>& grid_size,
            const Vector<int>& grid_cells,
            const bool generation_completed, Float& weight, int& bg_idx,
            const Camera& camera,
            const Float kbg,
            const Float* __restrict__ bg_tau_cum,
            const Float* __restrict__ z_lev_bg,
            const Float s_min,
            const Float s_min_bg)
    {
        ++photons_shot;
        if (!generation_completed)
        {
            //const Float i = (ij_cam % camera.nx) / Float(camera.nx) + Float(0.5) / Float(camera.nx);//(Float(0.5) * Float(camera.nx)) - Float(1.) + Float(0.5) / Float(camera.nx);
            //const Float j = Float(ij_cam/camera.nx) / (Float(0.5) * Float(camera.nx)) - Float(1.) + Float(0.5) / Float(camera.nx);
            const Float i = (Float(ij_cam % camera.nx) + rng())/ Float(camera.nx);//(Float(0.5) * Float(camera.nx)) - Float(1.) + Float(0.5) / Float(camera.nx);
            const Float j = (Float(ij_cam / camera.nx) + rng())/ Float(camera.nx);//(Float(0.5) * Float(camera.nx)) - Float(1.) + Float(0.5) / Float(camera.nx);

            // sunset
            // photon.position = {Float(1510.) + s_min,  Float(4710.) + s_min, Float(500.)+ s_min};
            // const Float photon_zenith = i * Float(.20) * M_PI;
            // const Float photon_azimuth = j * Float(2.) * M_PI;
            // const Float yaw = Float(-100.) / Float(180) * M_PI;
            // const Float pitch = Float(-25.) / Float(180) * M_PI;
            // const Float roll = Float(0.) / Float(180) * M_PI;


            // photon.position.x = Float(12004.) + s_min;
            // photon.position.y = Float(04.) + s_min; //Float4710);
            // photon.position.z = Float(54.)+ s_min;
            //photon.position.z = cam_data[1] + s_min; //Float(500.)+ s_min;

            photon.position = camera.position;

            const Float photon_zenith = i * Float(.5) * M_PI / camera.f_zoom;
            const Float photon_azimuth = j * Float(2.) * M_PI;
            const Vector<Float> dir_tmp = {sin(photon_zenith) * sin(photon_azimuth), sin(photon_zenith) * cos(photon_azimuth), cos(photon_zenith)};

            photon.direction.x = dot(camera.mx,  dir_tmp);
            photon.direction.y = dot(camera.my,  dir_tmp);
            photon.direction.z = dot(camera.mz,  dir_tmp) * Float(-1);

            // const Float yaw = Float(90.) / Float(180) * M_PI;
            // const Float pitch = Float(-90) / Float(180) * M_PI;
            // const Float roll = Float(0.) / Float(180) * M_PI;
            // const Float dir_x = cos(yaw)*sin(pitch)*photon.direction.x +
            //                     (cos(yaw)*cos(pitch)*sin(roll)-sin(yaw)*cos(roll))*photon.direction.y +
            //                     (cos(yaw)*cos(pitch)*cos(roll)+sin(yaw)*sin(roll)) * photon.direction.z;
            // const Float dir_y = sin(yaw)*sin(pitch)*photon.direction.x +
            //                     (sin(yaw)*cos(pitch)*sin(roll)+cos(yaw)*cos(roll))*photon.direction.y +
            //                     (sin(yaw)*cos(pitch)*cos(roll)-cos(yaw)*sin(roll)) * photon.direction.z;
            // const Float dir_z = -cos(pitch)*photon.direction.x +
            //                     sin(pitch)*sin(roll) * photon.direction.y +
            //                     sin(pitch)*cos(roll) * photon.direction.z;
            // photon.direction = {dir_x,dir_y,-dir_z};

            photon.kind = Photon_kind::Direct;
            weight = 1;
            bg_idx = 0;

            for (int i=0; i<kbg; ++i)
            {
                if (photon.position.z > z_lev_bg[i]) bg_idx = i;
            }

            if ( (dot(photon.direction, sun_direction) > cos_half_angle_app) )
            {
                const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                           k_null_grid,k_ext,
                                           bg_tau_cum, z_lev_bg, bg_idx,
                                           kn_grid, kn_grid_d, grid_d,
                                           grid_size, grid_cells,
                                           s_min, s_min_bg);

               atomicAdd(&camera_count[ij_cam], weight * trans_sun);
            }
            atomicAdd(&camera_shot[ij_cam], Float(1.));
        }
    }

    __device__
    inline Float probability_from_sun(
            Photon photon, Vector<Float> sun_direction, const Float solid_angle, const Float g, const Phase_kind kind)
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
        const Float* __restrict__ k_ext, const Optics_scat* __restrict__ scat_asy,
        const Float* __restrict__ k_ext_bg, const Optics_scat* __restrict__ scat_asy_bg,
        const Float* __restrict__ z_lev_bg,
        const Float* __restrict__ surface_albedo,
        const Float* __restrict__ land_use_map,
        const Float mu,
        const Vector<Float> grid_size, const Vector<Float> grid_d,
        const Vector<int> grid_cells, const Vector<int> kn_grid,
        const Vector<Float> sun_direction, const Camera camera,
        const int kbg)
{
    //const Phase_kind surface_kind = Phase_kind::Lambertian;
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    Vector<Float> normal_direction = {0, 0, 1};

    //const Float azimuth_cam = cam_data[0];
    //const Vector<Float> cam = {sin(zenith_cam) * sin(azimuth_cam), sin(zenith_cam) * cos(azimuth_cam), cos(zenith_cam) };
    //const Vector<Float> axis_h = normalize(cross(cam, upward_cam));
    //const Vector<Float> axis_v = normalize(cross(cam, axis_h));
    //const Vector<Float> axis_z = cam / tan(fov/Float(2.));

    extern __shared__ Float bg_tau_cum[];
    Float bg_tau = Float(0.);
    for (int k=kbg-1; k >= 0; --k)
    {
        bg_tau += k_ext_bg[k] * abs(z_lev_bg[k+1]-z_lev_bg[k]) / mu;
        bg_tau_cum[k] = bg_tau;
    }
    const Float bg_transmissivity = exp(-bg_tau_cum[0]);

    const Vector<Float> kn_grid_d = grid_size / kn_grid;
    //const Float kgrid_x = x_size/Float(ngrid_x);
    //const Float kgrid_y = y_size/Float(ngrid_y);
    //const Float kgrid_z = z_size/Float(ngrid_z);
    const Float z_top = z_lev_bg[kbg];

    Random_number_generator<Float> rng(n);

    const Float s_min = max(max(grid_size.z, grid_size.x), grid_size.y) * Float_epsilon;
    const Float s_min_bg = max(max(grid_size.x, grid_size.y), z_top) * Float_epsilon;

    const int pixels_per_thread = camera.nx * camera.ny / (bw_kernel_grid * bw_kernel_block);
    const int photons_per_pixel = photons_to_shoot / pixels_per_thread ;

    while (counter[0] < camera.nx*camera.ny)
    {
        const int ij_cam = atomicAdd(&counter[0], 1);

        if (ij_cam >= camera.nx*camera.ny)
            return;

        const int i = ij_cam % camera.nx;
        const int j = ij_cam / camera.nx;

        const bool completed = false;
        Int photons_shot = Atomic_reduce_const;
        Float weight;
        int bg_idx;

        Photon photon;

        reset_photon(
                photon, camera_count,camera_shot, photons_shot,
                ij_cam, n, rng, sun_direction,
                k_null_grid, k_ext,
                kn_grid, kn_grid_d, grid_d,
                grid_size, grid_cells,
                completed, weight, bg_idx,
                camera,
                kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);

        Float tau;
        Float d_max = Float(0.);
        Float k_ext_null;
        bool transition = false;
        int i_n, j_n, k_n, ijk_n;

        while (photons_shot < photons_per_pixel)
        {
            const bool photon_generation_completed = (photons_shot == photons_per_pixel - 1);

            if (!transition)
            {
                tau = sample_tau(rng());
            }
            transition = false;

            // 1D raytracing between TOD and TOA?
            if (photon.position.z > grid_size.z)
            {
                const Float dn = max(Float_epsilon, tau / k_ext_bg[bg_idx]);
                d_max = abs( (photon.direction.z>0) ? (z_lev_bg[bg_idx+1] - photon.position.z) / photon.direction.z : (z_lev_bg[bg_idx] - photon.position.z) / photon.direction.z );
                if (dn >= d_max)
                {
                    photon.position.z = (photon.direction.z > 0) ? z_lev_bg[bg_idx+1] + s_min_bg : z_lev_bg[bg_idx] - s_min_bg;
                    photon.position.y += photon.direction.y * d_max;
                    photon.position.x += photon.direction.x * d_max;

                    // move to actual grid: reduce tau and set next position
                    if (photon.position.z <= grid_size.z + s_min_bg)
                    {
                        tau -= k_ext_bg[bg_idx] * (d_max + s_min_bg);
                        photon.position.z = grid_size.z - s_min;
                        d_max = Float(0.);
                        transition=true;

                        // Cyclic boundary condition in x.
                        photon.position.x = fmod(photon.position.x, grid_size.x);
                        if (photon.position.x < Float(0.))
                            photon.position.x += grid_size.x;

                        // Cyclic boundary condition in y.
                        photon.position.y = fmod(photon.position.y, grid_size.y);
                        if (photon.position.y < Float(0.))
                            photon.position.y += grid_size.y;
                    }
                    else if (photon.position.z >= z_top)
                    {
                        // Leaving top-of-domain
                        d_max = Float(0.);
                        reset_photon(
                                photon, camera_count,camera_shot, photons_shot,
                                ij_cam, n, rng, sun_direction,
                                k_null_grid, k_ext,
                                kn_grid, kn_grid_d, grid_d,
                                grid_size, grid_cells,
                                completed, weight, bg_idx,
                                camera,
                                kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                    }
                    else
                    {
                        // just move to next grid
                        transition = true;
                        tau -= k_ext_bg[bg_idx] * (d_max + s_min_bg);

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
                    const Float k_sca_bg_tot = scat_asy_bg[bg_idx].k_sca_gas + scat_asy_bg[bg_idx].k_sca_cld + scat_asy_bg[bg_idx].k_sca_aer;
                    const Float ssa_bg_tot = k_sca_bg_tot / k_ext_bg[bg_idx];

                    // Update weights (see Iwabuchi 2006: https://doi.org/10.1175/JAS3755.1)
                    weight *= ssa_bg_tot;
                    if (weight < w_thres)
                        weight = (rng() > weight) ? Float(0.) : Float(1.);

                    // only with nonzero weight continue ray tracing, else start new ray
                    if (weight > Float(0.))
                    {
                        // find scatter type: 0 = gas, 1 = cloud, 2 = aerosol
                        const Float scatter_rng = rng();
                        const int scatter_type = scatter_rng < (scat_asy_bg[bg_idx].k_sca_aer/k_sca_bg_tot) ? 2 :
                                                 scatter_rng < ((scat_asy_bg[bg_idx].k_sca_aer+scat_asy_bg[bg_idx].k_sca_cld)/k_sca_bg_tot) ? 1 : 0;
                        Float g;
                        switch (scatter_type)
                        {
                            case 0:
                                g = Float(0.);
                                break;
                            case 1:
                                g = min(Float(1.) - Float_epsilon, scat_asy_bg[bg_idx].asy_cld);
                                break;
                            case 2:
                                g = min(Float(1.) - Float_epsilon, scat_asy_bg[bg_idx].asy_aer);
                                break;
                        }
                        const Float cos_scat = (scatter_type == 0) ? rayleigh(rng()) : henyey(g, rng());
                        const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                        // direct contribution
                        const Phase_kind kind = (scatter_type==0) ? Phase_kind::Rayleigh : Phase_kind::HG;
                        const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, g, kind);
                        const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                    k_null_grid,k_ext,
                                                    bg_tau_cum, z_lev_bg, bg_idx,
                                                    kn_grid, kn_grid_d, grid_d,
                                                    grid_size, grid_cells,
                                                    s_min, s_min_bg);
                        atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);


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
                    else
                    {
                        d_max = Float(0.);
                        reset_photon(
                                photon, camera_count,camera_shot, photons_shot,
                                ij_cam, n, rng, sun_direction,
                                k_null_grid, k_ext,
                                kn_grid, kn_grid_d, grid_d,
                                grid_size, grid_cells,
                                photon_generation_completed, weight, bg_idx,
                                camera,
                                kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                    }
                }
            }
            // we reached the 'dynamical' domain, now things get interesting
            else
            {
                // if d_max is zero, find current grid and maximum distance
                if (d_max == Float(0.))
                {
                    i_n = float_to_int(photon.position.x, kn_grid_d.x, kn_grid.x);
                    j_n = float_to_int(photon.position.y, kn_grid_d.y, kn_grid.y);
                    k_n = float_to_int(photon.position.z, kn_grid_d.z, kn_grid.z);
                    const Float sx = abs((photon.direction.x > 0) ? ((i_n+1) * kn_grid_d.x - photon.position.x)/photon.direction.x : (i_n*kn_grid_d.x - photon.position.x)/photon.direction.x);
                    const Float sy = abs((photon.direction.y > 0) ? ((j_n+1) * kn_grid_d.y - photon.position.y)/photon.direction.y : (j_n*kn_grid_d.y - photon.position.y)/photon.direction.y);
                    const Float sz = abs((photon.direction.z > 0) ? ((k_n+1) * kn_grid_d.z - photon.position.z)/photon.direction.z : (k_n*kn_grid_d.z - photon.position.z)/photon.direction.z);
                    d_max = min(sx, min(sy, sz));
                    ijk_n = i_n + j_n*kn_grid.x + k_n*kn_grid.y*kn_grid.x;
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
                        const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                        const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                        const int ij = i + j*grid_cells.x;
                        d_max = Float(0.);

                        // Update weights and add upward surface flux
                        const Float local_albedo =  surface_albedo[ij];
                        weight *= local_albedo;

                        const Phase_kind surface_kind = (land_use_map[ij] == 0) ? Phase_kind::Specular : Phase_kind::Lambertian;

                        // SUN SCATTERING GOES HERE
                        const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, Float(0.), (photon.kind == Photon_kind::Direct) ? surface_kind : Phase_kind::Lambertian);
                        const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                    k_null_grid,k_ext,
                                                    bg_tau_cum, z_lev_bg, bg_idx,
                                                    kn_grid, kn_grid_d, grid_d,
                                                    grid_size, grid_cells,
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
                                    kn_grid, kn_grid_d, grid_d,
                                    grid_size, grid_cells,
                                    photon_generation_completed, weight, bg_idx,
                                    camera,
                                    kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                        }
                    }

                    // TOD exit
                    else if (photon.position.z >= grid_size.z)
                    {
                        photon.position.z = grid_size.z + s_min_bg;
                        tau -= d_max * k_ext_null;
                        bg_idx = 0;
                        d_max = Float(0.);
                        transition = true;

                    }
                    // regular cell crossing: adjust tau and apply periodic BC
                    else
                    {
                        photon.position.x += photon.direction.x>0 ? s_min : -s_min;
                        photon.position.y += photon.direction.y>0 ? s_min : -s_min;
                        photon.position.z += photon.direction.z>0 ? s_min : -s_min;

                        // Cyclic boundary condition in x.
                        photon.position.x = fmod(photon.position.x, grid_size.x);
                        if (photon.position.x < Float(0.))
                            photon.position.x += grid_size.x;

                        // Cyclic boundary condition in y.
                        photon.position.y = fmod(photon.position.y, grid_size.y);
                        if (photon.position.y < Float(0.))
                            photon.position.y += grid_size.y;

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

                    photon.position.x = (dx > 0) ? min(photon.position.x + dx, (i_n+1) * kn_grid_d.x - s_min) : max(photon.position.x + dx, (i_n) * kn_grid_d.x + s_min);
                    photon.position.y = (dy > 0) ? min(photon.position.y + dy, (j_n+1) * kn_grid_d.y - s_min) : max(photon.position.y + dy, (j_n) * kn_grid_d.y + s_min);
                    photon.position.z = (dz > 0) ? min(photon.position.z + dz, (k_n+1) * kn_grid_d.z - s_min) : max(photon.position.z + dz, (k_n) * kn_grid_d.z + s_min);

                    // Calculate the 3D index.
                    const int i = float_to_int(photon.position.x, grid_d.x, grid_cells.x);
                    const int j = float_to_int(photon.position.y, grid_d.y, grid_cells.y);
                    const int k = float_to_int(photon.position.z, grid_d.z, grid_cells.z);
                    const int ijk = i + j*grid_cells.x + k*grid_cells.x*grid_cells.y;


                    // Compute probability not being absorbed and store weighted absorption probability
                    const Float k_sca_tot = scat_asy[ijk].k_sca_gas + scat_asy[ijk].k_sca_cld + scat_asy[ijk].k_sca_aer;
                    const Float ssa_tot = k_sca_tot / k_ext[ijk];
                    const Float f_no_abs = Float(1.) - (Float(1.) - ssa_tot) * (k_ext[ijk]/k_ext_null);

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
                            const Float cos_scat = (scatter_type == 0) ? rayleigh(rng()) : henyey(g, rng());
                            const Float sin_scat = max(Float(0.), sqrt(Float(1.) - cos_scat*cos_scat + Float_epsilon));

                            // SUN SCATTERING GOES HERE
                            const Phase_kind kind = (scatter_type==0) ? Phase_kind::Rayleigh : Phase_kind::HG;
                            const Float p_sun = probability_from_sun(photon, sun_direction, solid_angle, g, kind);
                            const Float trans_sun = transmission_direct_sun(photon,n,rng,sun_direction,
                                                        k_null_grid,k_ext,
                                                        bg_tau_cum, z_lev_bg, bg_idx,
                                                        kn_grid, kn_grid_d, grid_d,
                                                        grid_size, grid_cells,
                                                        s_min, s_min_bg);
                            atomicAdd(&camera_count[ij_cam], weight * p_sun * trans_sun);

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
                                photon, camera_count,camera_shot, photons_shot,
                                ij_cam, n, rng, sun_direction,
                                k_null_grid, k_ext,
                                kn_grid, kn_grid_d, grid_d,
                                grid_size, grid_cells,
                                photon_generation_completed, weight, bg_idx,
                                camera,
                                kbg, bg_tau_cum, z_lev_bg, s_min, s_min_bg);
                    }
                }
            }
        }
    }
}

