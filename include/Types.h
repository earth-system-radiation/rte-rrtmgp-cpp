#ifndef TYPES_H
#define TYPES_H

#include <map>

#ifdef RTE_RRTMGP_USE_CBOOL
using Bool = signed char;
using BOOL_TYPE = signed char;
#else
using Bool = int;
using BOOL_TYPE = int;
#endif

#ifdef RTE_RRTMGP_SINGLE_PRECISION
using Float = float;
using FLOAT_TYPE = float;
#else
using Float = double;
using FLOAT_TYPE = double;
#endif

#endif
