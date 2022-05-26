/*
 * This file is a stand-alone executable developed for the
 * testing of the C++ interface to the RTE+RRTMGP radiation code.
 *
 * It is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/algorithm/string.hpp>
#include <chrono>
#include <iomanip>
#include <cuda_profiler_api.h>


#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "raytracer_kernels.h"
#include "Radiation_solver_rt.h"
#include "Gas_concs.h"
#include "Types.h"
#include "Mem_pool_gpu.h"


bool parse_command_line_options(
        std::map<std::string, std::pair<bool, std::string>>& command_line_options,
        Int& ray_count_exponent,
        int argc, char** argv)
{
    for (int i=1; i<argc; ++i)
    {
        std::string argument(argv[i]);
        boost::trim(argument);

        if (argument == "-h" || argument == "--help")
        {
            Status::print_message("Possible usage:");
            for (const auto& clo : command_line_options)
            {
                std::ostringstream ss;
                ss << std::left << std::setw(30) << ("--" + clo.first);
                ss << clo.second.second << std::endl;
                Status::print_message(ss);
            }
            return true;
        }

        //check if option is integer n (2**n rays)
        if (std::isdigit(argument[0]))
        {
            if (argument.size() > 1)
            {
                for (int i=1; i<argument.size(); ++i)
                {
                    if (!std::isdigit(argument[i]))
                    {
                        std::string error = argument + " is an illegal command line option.";
                        throw std::runtime_error(error);
                    }

                }
            }
            ray_count_exponent = Int(std::stoi(argv[i]));
        }
        else
        {
            // Check if option starts with --
            if (argument[0] != '-' || argument[1] != '-')
            {
                std::string error = argument + " is an illegal command line option.";
                throw std::runtime_error(error);
            }
            else
                argument.erase(0, 2);

            // Check if option has prefix no-
            bool enable = true;
            if (argument[0] == 'n' && argument[1] == 'o' && argument[2] == '-')
            {
                enable = false;
                argument.erase(0, 3);
            }

            if (command_line_options.find(argument) == command_line_options.end())
            {
                std::string error = argument + " is an illegal command line option.";
                throw std::runtime_error(error);
            }
            else
                command_line_options.at(argument).first = enable;
        }
    }

    return false;
}


void print_command_line_options(
        const std::map<std::string, std::pair<bool, std::string>>& command_line_options)
{
    Status::print_message("Solver settings:");
    for (const auto& option : command_line_options)
    {
        std::ostringstream ss;
        ss << std::left << std::setw(20) << (option.first);
        ss << " = " << std::boolalpha << option.second.first << std::endl;
        Status::print_message(ss);
    }
}


void solve_radiation(int argc, char** argv)
{
    Status::print_message("###### Starting RTE+RRTMGP solver ######");

    ////// FLOW CONTROL SWITCHES //////
    // Parse the command line options.
    std::map<std::string, std::pair<bool, std::string>> command_line_options {
        {"shortwave"        , { true,  "Enable computation of shortwave radiation."}},
        {"longwave"         , { true,  "Enable computation of longwave radiation." }},
        {"fluxes"           , { true,  "Enable computation of fluxes."             }},
        {"raytracing"       , { false, "Use raytracing for flux computation."      }},
        {"cloud-optics"     , { false, "Enable cloud optics."                      }},
        {"output-optical"   , { false, "Enable output of optical properties."      }},
        {"output-bnd-fluxes", { false, "Enable output of band fluxes."             }} };

    Int ray_count_exponent = 22;

    if (parse_command_line_options(command_line_options, ray_count_exponent, argc, argv))
        return;
    

    const bool switch_shortwave         = command_line_options.at("shortwave"        ).first;
    const bool switch_longwave          = command_line_options.at("longwave"         ).first;
    const bool switch_fluxes            = command_line_options.at("fluxes"           ).first;
    const bool switch_raytracing        = command_line_options.at("raytracing"       ).first;
    const bool switch_cloud_optics      = command_line_options.at("cloud-optics"     ).first;
    const bool switch_output_optical    = command_line_options.at("output-optical"   ).first;
    const bool switch_output_bnd_fluxes = command_line_options.at("output-bnd-fluxes").first;
    
    // Print the options to the screen.
    print_command_line_options(command_line_options);

    Int ray_count;
    if (switch_raytracing)
    {
        ray_count = pow(2,ray_count_exponent);
        if (ray_count < block_size*grid_size)
        {
            std::string error = "Cannot shoot " + std::to_string(ray_count) + " rays with current block/grid sizes.";
            throw std::runtime_error(error);
        }
        else
            Status::print_message("Using "+ std::to_string(Int(pow(2, ray_count_exponent))) + " rays");
    }


    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc("rte_rrtmgp_input.nc", Netcdf_mode::Read);

    const int nx = input_nc.get_dimension_size("x");
    const int ny = input_nc.get_dimension_size("y");
    const int nz = input_nc.get_dimension_size("z");


    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.
    Status::print_message("Preparing NetCDF output file.");

    Netcdf_file output_nc("rte_rrtmgp_output.nc", Netcdf_mode::Create);
    output_nc.add_dimension("x", nx);
    output_nc.add_dimension("y", ny);
    output_nc.add_dimension("z", nz);


    ////// RUN THE SHORTWAVE SOLVER //////
    if (switch_shortwave)
    {
        // Initialize the solver.
        Status::print_message("Initializing the shortwave solver.");

        // Solve the radiation.
        Status::print_message("Solving the shortwave radiation.");

        auto run_solver = [&]()
        {
            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);
            // do something.

            /*
            raytracer.trace_rays(
                    ray_count,
                    nx, ny, nz,
                    dx_grid, dy_grid, dz_grid,
                    dynamic_cast<Optical_props_2str_rt&>(*optical_props),
                    dynamic_cast<Optical_props_2str_rt&>(*cloud_optical_props),
                    sfc_alb_dir, zenith_angle, 
                    azimuth_angle,
                    tod_dir_diff({1}),
                    tod_dir_diff({2}),
                    (*fluxes).get_flux_tod_up(),
                    (*fluxes).get_flux_sfc_dir(),
                    (*fluxes).get_flux_sfc_dif(),
                    (*fluxes).get_flux_sfc_up(),
                    (*fluxes).get_flux_abs_dir(),
                    (*fluxes).get_flux_abs_dif());
                    */

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration shortwave solver: " + std::to_string(duration) + " (ms)");
        };

        // Tuning step;
        run_solver();

        // Profiling step;
        cudaProfilerStart();
        run_solver();
        cudaProfilerStop();

        // Store the output.
        Status::print_message("Storing the shortwave output.");
    }

    Status::print_message("###### Finished RTE+RRTMGP solver ######");
}


int main(int argc, char** argv)
{
    try
    {
        solve_radiation(argc, argv);
    }

    // Catch any exceptions and return 1.
    catch (const std::exception& e)
    {
        std::string error = "EXCEPTION: " + std::string(e.what());
        Status::print_message(error);
        return 1;
    }
    catch (...)
    {
        Status::print_message("UNHANDLED EXCEPTION!");
        return 1;
    }

    // Return 0 in case of normal exit.
    return 0;
}
