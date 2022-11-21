#ifndef RAYTRACER_FUNCTIONS_H
#define RAYTRACER_FUNCTIONS_H
#include "Types.h"
#include <curand_kernel.h>

namespace Raytracer_functions
{

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
    Vector<T> operator+(const Vector<T> v, const Float s) { return Vector<T>{s+v.x, s+v.y, s+v.z}; }
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

    struct Optics_scat
    {
        Float k_sca_gas;
        Float k_sca_cld;
        Float k_sca_aer;
        Float asy_cld;
        Float asy_aer;
    };

    enum class Photon_kind { Direct, Diffuse };

    struct Photon
    {
        Vector<Float>position;
        Vector<Float>direction;
        Photon_kind kind;
    };

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

    __device__
    inline Float pow2(const Float d) { return d*d; }

    __device__
    inline Float rayleigh(const Float random_number)
    {
        const Float q = Float(4.)*random_number - Float(2.);
        const Float d = Float(1.) + pow2(q);
        const Float u = pow(-q + sqrt(d), Float(1./3.));
        return u - Float(1.)/u;
    }


    __device__
    inline Float henyey(const Float g, const Float random_number)
    {
        const Float a = pow2(Float(1.) - pow2(g));
        const Float b = Float(2.)*g*pow2(Float(2.)*random_number*g + Float(1.) - g);
        const Float c = -g/Float(2.) - Float(1.)/(Float(2.)*g);
        return Float(-1.)*(a/b) - c;
    }

    __device__
    inline Float sample_tau(const Float random_number)
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
    __device__
    inline void write_photon_out(Float* field_out, const Float w)
    {
        #ifdef __CUDACC__
        atomicAdd(field_out, w);
        #endif
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
    __device__ inline double Random_number_generator<double>::operator()()
    {
        return 1. - curand_uniform_double(&state);
    }


    template<>
    __device__ inline float Random_number_generator<float>::operator()()
    {
        return 1.f - curand_uniform(&state);
    }

}

#endif
