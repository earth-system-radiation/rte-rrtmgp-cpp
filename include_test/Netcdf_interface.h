/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/RobertPincus/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/microhh/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#ifndef NETCDF_INTERFACE_H
#define NETCDF_INTERFACE_H

#include <iostream>
#include <map>
#include <vector>
#include <numeric>
#include <netcdf.h>
#include "Status.h"

enum class Netcdf_mode { Create, Read, Write };

class Netcdf_handle;
class Netcdf_group;

template<typename T>
class Netcdf_variable
{
    public:
        Netcdf_variable(Netcdf_handle&, const int, const std::vector<int>&);
        Netcdf_variable(const Netcdf_variable&) = default;
        void insert(const std::vector<T>&, const std::vector<int>);
        void insert(const std::vector<T>&, const std::vector<int>, const std::vector<int>);
        void insert(const T, const std::vector<int>);
        const std::vector<int> get_dim_sizes() { return dim_sizes; }
        void add_attribute(const std::string&, const std::string&);
        void add_attribute(const std::string&, const double);
        void add_attribute(const std::string&, const float);

    private:
        Netcdf_handle& nc_file;
        const int var_id;
        const std::vector<int> dim_sizes;
};

class Netcdf_handle
{
    public:
        Netcdf_handle();
        void add_dimension(const std::string&, const int dim_size = NC_UNLIMITED);

        Netcdf_group add_group(const std::string&);
        Netcdf_group get_group(const std::string&);

        int get_dimension_size(const std::string&);

        std::map<std::string, int> get_variable_dimensions(const std::string&);

        bool variable_exists(const std::string&);

        template<typename T>
        Netcdf_variable<T> add_variable(
                const std::string&,
                const std::vector<std::string>);

        template<typename T>
        T get_variable(
            const std::string&);

        template<typename T>
        std::vector<T> get_variable(
            const std::string&,
            const std::vector<int>&);

        template<typename T>
        void get_variable(
                std::vector<T>&,
                const std::string&,
                const std::vector<int>&,
                const std::vector<int>&);

        template<typename T>
        void insert(
                const std::vector<T>&,
                const int var_id,
                const std::vector<int>&,
                const std::vector<int>&);

        template<typename T>
        void insert(
                const T,
                const int var_id,
                const std::vector<int>&,
                const std::vector<int>&);

        void add_attribute(
                const std::string&,
                const std::string&,
                const int);

        void add_attribute(
                const std::string&,
                const double,
                const int);

        void add_attribute(
                const std::string&,
                const float,
                const int);

    protected:
        int ncid;
        int root_ncid;
        std::map<std::string, int> dims;
        int record_counter;
};

class Netcdf_file : public Netcdf_handle
{
    public:
        Netcdf_file(const std::string&, Netcdf_mode);
        ~Netcdf_file();

        void sync();
};

class Netcdf_group : public Netcdf_handle
{
    public:
        Netcdf_group(const int, const int);
};
#endif
