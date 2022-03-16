#ifndef TYPES_H
#define TYPES_H

#include <map>

#ifdef RTE_RRTMGP_USE_CBOOL
using Bool = signed char;
#else
using Bool = int;
#endif

#ifdef RTE_RRTMGP_SINGLE_PRECISION
using Float = float;
#else
using Float = double;
#endif

#endif
