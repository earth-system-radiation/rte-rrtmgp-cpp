import netCDF4 as nc
import numpy as np

ref_nc = nc.Dataset("ref.nc", "r")
data_nc = nc.Dataset("rte_rrtmgp_output.nc", "r")

def compare(name):
    ref = ref_nc.variables[name][:]
    data = data_nc.variables[name][:]
    print(name, abs(ref - data).max())

compare("lw_tau")
compare("lay_source")
compare("lev_source_inc")
compare("lev_source_dec")
compare("sfc_source")

compare("sw_tau")
compare("ssa")
compare("g")
compare("toa_source")

# compare("lw_flux_up")
# compare("lw_flux_dn")
# compare("lw_flux_net")
# 
# compare("sw_flux_up")
# compare("sw_flux_dn")
# compare("sw_flux_dn_dir")
# compare("sw_flux_net")
