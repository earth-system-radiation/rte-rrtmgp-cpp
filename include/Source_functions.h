#ifndef SOURCE_FUNCTIONS_H
#define SOURCE_FUNCTIONS_H

#include "Array.h"
#include "Optical_props.h"

template<typename TF>
class Source_func_lw : public Optical_props<TF>
{
    public:
        Source_func_lw(
                const int n_col,
                const int n_lay,
                const Optical_props<TF>& optical_props) :
            Optical_props<TF>(optical_props),
            sfc_source({n_col, optical_props.get_ngpt()}),
            lay_source({n_col, n_lay, optical_props.get_ngpt()}),
            lev_source_inc({n_col, n_lay, optical_props.get_ngpt()}),
            lev_source_dec({n_col, n_lay, optical_props.get_ngpt()})
        {}

        Array<TF,2>& get_sfc_source() { return sfc_source; }
        Array<TF,3>& get_lay_source() { return lay_source; }
        Array<TF,3>& get_lev_source_inc() { return lev_source_inc; }
        Array<TF,3>& get_lev_source_dec() { return lev_source_dec; }

    private:
        Array<TF,2> sfc_source;
        Array<TF,3> lay_source;
        Array<TF,3> lev_source_inc;
        Array<TF,3> lev_source_dec;
};
#endif
