# C++ interface of RTE+RRTMGP
This is a C++ interface to the Radiative Transfer for Energetics (RTE)
and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).

The original code is found at https://github.com/RobertPincus/rte-rrtmgp.

Contacts: Robert Pincus and Eli Mlawer
email: rrtmgp@aer.com

This C++ interface can be downloaded from https://github.com/microhh/rte-rrtmgp-cpp

Contact: Chiel van Heerwaarden
email: chiel.vanheerwaarden@wur.nl

Use and duplication is permitted under the terms of the
BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause

In order to check out the code including the `rte-rrtmgp` submodule, use:

    git clone --recurse-submodules https://github.com/microhh/rte-rrtmgp-cpp

In case you had already checked out the repository, use:

    git submodule update --init


# Some temporary instructions
In order to run the RCEMIP test:
1. Place the long wave coefficients file from the original RTE+RRTMGP repository as `coefficients_lw.nc` in the directory of the executable.
2. Place the short wave coefficients file from the original RTE+RRTMGP repository as `coefficients_sw.nc` in the directory of the executable.

In order to run the RTE+RRTMGP unit test with long wave:
1. Place either the long wave coefficients file from the original RTE+RRTMGP repository as `coefficients.nc` in the directory of the executable.
2. Place the `rrtmgp-lw-inputs-outputs-clear.nc` under the name `rrtmgp-inputs-outputs.nc` in the directory of the executable.

In order to run the RTE+RRTMGP unit test with short wave:
1. Place either the short wave coefficients file from the original RTE+RRTMGP repository as `coefficients.nc` in the directory of the executable.
2. Place the `rrtmgp-sw-inputs-outputs-clear.nc` under the name `rrtmgp-inputs-outputs.nc` in the directory of the executable.

