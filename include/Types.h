#ifndef TYPES_H
#define TYPES_H

#include <map>

#ifdef RTE_RRTMGP_USE_CBOOL
using BOOL_TYPE = signed char;
#else
using BOOL_TYPE = int;
#endif

#ifdef RTE_RRTMGP_SINGLE_PRECISION
using FLOAT_TYPE = float;
#else
using FLOAT_TYPE = double;
#endif

#ifdef __CUDACC__
using Tuner_map = std::map<std::string, std::pair<dim3, dim3>>;
#endif

#endif
