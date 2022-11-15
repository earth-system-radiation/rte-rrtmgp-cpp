#ifndef RAYTRACER_KERNELS_BW_H
#define RAYTRACER_KERNELS_BW_H
#include <curand_kernel.h>
#include "Types.h"



#ifdef RTE_RRTMGP_SINGLE_PRECISION
//using Float = float;
constexpr int bw_kernel_block= 512;
constexpr int bw_kernel_grid = 1024;
#else
//using Float = double;
constexpr int bw_kernel_block = 256;
constexpr int bw_kernel_grid = 256;
#endif
using Int = unsigned long long;
//constexpr int ngrid_x = 96;
//constexpr int ngrid_y = 96;
//constexpr int ngrid_z = 24;
constexpr Float k_null_gas_min = Float(1.e-3);

struct Optics_scat
{
    Float k_sca_gas;
    Float k_sca_cld;
    Float k_sca_aer;
    Float asy_cld;
    Float asy_aer;
};

struct Grid_knull
{
    Float k_max;
    Float k_min;
};

template<typename T>
struct Vector
{
    T x;
    T y;
    T z;
};

template<typename T> static inline __host__ __device__
Vector<T> operator*(const Vector<T> v, const Float s) { return Vector<T>{s*v.x, s*v.y, s*v.z}; }
template<typename T> static inline __host__ __device__
Vector<T> operator*(const Float s, const Vector<T> v) { return Vector<T>{s*v.x, s*v.y, s*v.z}; }
template<typename T> static inline __host__ __device__
Vector<T> operator-(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
template<typename T> static inline __host__ __device__
Vector<T> operator+(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }
template<typename T> static inline __host__ __device__
Vector<T> operator/(const Vector<T> v, const Float s) { return Vector<T>{v.x/s, v.y/s, v.z/s}; }
template<typename T> static inline __host__ __device__
Vector<T> operator/(const Float s, const Vector<T> v) { return Vector<T>{v.x/s, v.y/s, v.z/s}; }
template<typename T> static inline __host__ __device__
Vector<T> operator*(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x*v2.x, v1.y*v2.y, v1.z*v2.z}; }
template<typename T> static inline __host__ __device__
Vector<T> operator/(const Vector<T> v1, const Vector<T> v2) { return Vector<T>{v1.x/v2.x, v1.y/v2.y, v1.z/v2.z}; }

static inline __host__ __device__
Vector<Float> operator/(const Vector<Float> v1, const Vector<int> v2) { return Vector<Float>{v1.x/v2.x, v1.y/v2.y, v1.z/v2.z}; }
static inline __host__ __device__
Vector<Float> operator*(const Vector<Float> v1, const Vector<int> v2) { return Vector<Float>{v1.x*v2.x, v1.y*v2.y, v1.z*v2.z}; }

struct Camera
{
    Vector<Float> pos;

    // rotation matrix
    Vector<Float> mx;
    Vector<Float> my;
    Vector<Float> mz;

    Float f_zoom;
    void setup_rotation_matrix(const Float yaw, const Float pitch, const Float roll)
    {
        mx = {cos(yaw)*sin(pitch), (cos(yaw)*cos(pitch)*sin(roll)-sin(yaw)*cos(roll)), (cos(yaw)*cos(pitch)*cos(roll)+sin(yaw)*sin(roll))};
        my = {sin(yaw)*sin(pitch), (sin(yaw)*cos(pitch)*sin(roll)+cos(yaw)*cos(roll)), (sin(yaw)*cos(pitch)*cos(roll)-cos(yaw)*sin(roll))};
        mz = {-cos(pitch), sin(pitch)*sin(roll), sin(pitch)*cos(roll)};
    }

    // size of output arrays, either number of horizontal and vertical pixels, or number of zenith/azimuth angles of fisheye lens
    int ny = 1024;
    int nx = 1024;
};

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
        const Vector<Float> grid_size,
        const Vector<Float> grid_d,
        const Vector<int> grid_cells,
        const Vector<int> kn_grid,
        const Vector<Float> sun_direction,
        const Camera camera,
        const int nbg);
#endif
