#ifndef TUNER_H
#define TUNER_H

#include <iostream>
#include <iomanip>

template<class Func, class... Args>
std::tuple<dim3, dim3> tune_kernel(
        const std::string& kernel_name,
        dim3 problem_size,
        const std::vector<int>& ib, const std::vector<int>& jb, const std::vector<int>& kb, 
        Func&& f, Args&&... args)
{
    std::cout << "Tuning " << kernel_name << ": ";

    float fastest = std::numeric_limits<float>::max();
    dim3 fastest_block{1, 1, 1};
    dim3 fastest_grid{problem_size};

    for (const int k : kb)
        for (const int j : jb)
            for (const int i : ib)
            {
                dim3 block{i, j, k};
                dim3 grid{
                    problem_size.x/block.x + (problem_size.x%block.x > 0),
                    problem_size.y/block.y + (problem_size.y%block.y > 0),
                    problem_size.z/block.z + (problem_size.z%block.z > 0)};

                // Warmup...
                f<<<grid, block>>>(args...);

                cudaEvent_t start;
                cudaEvent_t stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                constexpr int n_samples = 8;

                cudaEventRecord(start, 0);
                for (int i=0; i<n_samples; ++i)
                    f<<<grid, block>>>(args...);
                cudaEventRecord(stop, 0);

                cudaEventSynchronize(stop);
                float duration = 0.f;
                cudaEventElapsedTime(&duration, start, stop);

                // Check whether kernel has succeeded.
                cudaError err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    // std::cout << "("
                    //     << std::setw(3) << i << ", " << std::setw(3) << j << ", " << std::setw(3) << k << ") "
                    //     << "FAILED! " << std::endl;
                }
                else
                {
                    if (duration < fastest)
                    {
                        fastest = duration;
                        fastest_grid = grid;
                        fastest_block = block;
                    }

                    // std::cout << "("
                    //     << std::setw(3) << i << ", " << std::setw(3) << j << ", " << std::setw(3) << k << ") "
                    //     << std::setprecision(5) << duration/n_samples << " (ns)" << std::endl;
                }
            }

     std::cout << "(" 
         << fastest_block.x << ", "
         << fastest_block.y << ", "
         << fastest_block.z << ")" << std::endl;

     return {fastest_grid, fastest_block};
}


template<class Func, int I, int J, class... Args>
void run_inner(
        dim3 problem_size, dim3& fastest_grid, dim3& fastest_block, float& fastest, Args... args)
{
    dim3 block{I, J, 1};
    dim3 grid{
        problem_size.x/block.x + (problem_size.x%block.x > 0),
        problem_size.y/block.y + (problem_size.y%block.y > 0),
        problem_size.z/block.z + (problem_size.z%block.z > 0)};
    
    // Warmup...
    Func::template launch<I, J>(grid, block, args...);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    constexpr int n_samples = 8;
    
    cudaEventRecord(start, 0);
    for (int i=0; i<n_samples; ++i)
        Func::template launch<I, J>(grid, block, args...);
    cudaEventRecord(stop, 0);
    
    cudaEventSynchronize(stop);
    float duration = 0.f;
    cudaEventElapsedTime(&duration, start, stop);
    
    // Check whether kernel has succeeded.
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "("
            << std::setw(3) << I << ", " << std::setw(3) << J << ", " << std::setw(3) << 1 << ") "
            << "FAILED! " << std::endl;
    }
    else
    {
        if (duration < fastest)
        {
            fastest = duration;
            fastest_grid = grid;
            fastest_block = block;
        }

        std::cout << "("
            << std::setw(3) << I << ", " << std::setw(3) << J << ", " << std::setw(3) << 1 << ") "
            << std::setprecision(5) << duration/n_samples << " (ns)" << std::endl;
    }
}

template<class Func, int I, int... Js, class... Args>
void run_outer(
        std::integer_sequence<int, Js...> js,
        dim3 problem_size, dim3& fastest_grid, dim3& fastest_block, float& fastest, Args... args)
{
    (run_inner<Func, I, Js>(problem_size, fastest_grid, fastest_block, fastest, args...), ...);
}

template<class Func, class... Args, int... Is, int... Js>
std::tuple<dim3, dim3> tune_kernel_compile_time(
        const std::string& kernel_name,
        dim3 problem_size,
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js,
        Args&&... args)
{
    std::cout << "Tuning " << kernel_name << ": ";

    float fastest = std::numeric_limits<float>::max();
    dim3 fastest_block{1, 1, 1};
    dim3 fastest_grid{problem_size};

    (run_outer<Func, Is>(js, problem_size, fastest_grid, fastest_block, fastest, args...), ...);

    std::cout << "(" 
        << fastest_block.x << ", "
        << fastest_block.y << ", "
        << fastest_block.z << ")" << std::endl;

    return {fastest_grid, fastest_block};
}
#endif
