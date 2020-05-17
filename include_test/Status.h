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

#ifndef STATUS_H
#define STATUS_H

#include <sstream>
#include <string>
#include <iostream>

namespace Status
{
    void print_message(const std::ostringstream& ss)
    {
        std::cout << ss.str();
    }
    
    void print_message(const std::string& s)
    {
        std::cout << s << std::endl;
    }
    
    void print_warning(const std::ostringstream& ss)
    {
        std::cout << "WARNING: " << ss.str();
    }
    
    void print_warning(const std::string& s)
    {
        std::cout << "WARNING: " << s << std::endl;
    }
    
    void print_error(const std::ostringstream& ss)
    {
        std::cout << "ERROR: " << ss.str();
        std::cerr << "ERROR: " << ss.str();
    }
    
    void print_error(const std::string& s)
    {
        std::cout << "ERROR: " << s << std::endl;
        std::cerr << "ERROR: " << s << std::endl;
    }
}
#endif
