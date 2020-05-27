import numpy as np
import netCDF4 as nc

# Settings
float_type = 'f8'

# Save all the input data to NetCDF
nc_file = nc.Dataset('rte_rrtmgp_output.nc', mode='r')

nc_file_rld = nc.Dataset('rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
nc_file_rlu = nc.Dataset('rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
nc_file_rsd = nc.Dataset('rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')
nc_file_rsu = nc.Dataset('rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc', mode='a')

nc_file_rld.variables['rld'][0,:,:] = nc_file.variables['lw_flux_dn'][:,:].transpose()
nc_file_rlu.variables['rlu'][0,:,:] = nc_file.variables['lw_flux_up'][:,:].transpose()
nc_file_rsd.variables['rsd'][0,:,:] = nc_file.variables['sw_flux_dn'][:,:].transpose()
nc_file_rsu.variables['rsu'][0,:,:] = nc_file.variables['sw_flux_up'][:,:].transpose()

nc_file.close()
nc_file_rld.close()
nc_file_rlu.close()
nc_file_rsd.close()
nc_file_rsu.close()
