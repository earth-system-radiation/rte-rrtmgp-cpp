//
// cuda version aerosol optics, see src/Aerosol_optics.cpp by Mirjam Tijhuis
//

#ifndef AEROSOL_OPTICS_RT_H
#define AEROSOL_OPTICS_RT_H

#include "Array.h"
#include "Optical_props_rt.h"
#include "Types.h"


// Forward declarations.
class Optical_props_rt;

#ifdef USECUDA
class Aerosol_optics_rt : public Optical_props_rt
{
    public:
        Aerosol_optics_rt(
                Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
                const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
                const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic);


        void aerosol_optics(
                const int ibnd,
                const Array_gpu<Float,1>& aermr01, const Array_gpu<Float,1>& aermr02, const Array_gpu<Float,1>& aermr03,
                const Array_gpu<Float,1>& aermr04, const Array_gpu<Float,1>& aermr05, const Array_gpu<Float,1>& aermr06,
                const Array_gpu<Float,1>& aermr07, const Array_gpu<Float,1>& aermr08, const Array_gpu<Float,1>& aermr09,
                const Array_gpu<Float,1>& aermr10, const Array_gpu<Float,1>& aermr11,
                const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
                Optical_props_2str_rt& optical_props);

    private:
        // Lookup table coefficients
        Array<Float,2> mext_phobic;
        Array<Float,2> ssa_phobic;
        Array<Float,2> g_phobic;

        Array<Float,3> mext_philic;
        Array<Float,3> ssa_philic;
        Array<Float,3> g_philic;

        Array<Float,1> rh_upper;

        // gpu versions
        Array_gpu<Float,2> mext_phobic_gpu;
        Array_gpu<Float,2> ssa_phobic_gpu;
        Array_gpu<Float,2> g_phobic_gpu;

        Array_gpu<Float,3> mext_philic_gpu;
        Array_gpu<Float,3> ssa_philic_gpu;
        Array_gpu<Float,3> g_philic_gpu;

        Array_gpu<Float,1> rh_upper_gpu;

};
#endif

#endif
