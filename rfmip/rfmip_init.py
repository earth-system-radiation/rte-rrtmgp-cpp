import numpy as np
import netCDF4 as nc

# Settings
float_type = 'f8'

expt = 0 # Do not allow cross experiment runs
band_lw = 16
band_sw = 14

# Save all the input data to NetCDF
nc_file = nc.Dataset('rte_rrtmgp_input.nc', mode='w', datamodel='NETCDF4', clobber=True)
nc_file_rfmip = nc.Dataset('multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', mode='r', datamodel='NETCDF4', clobber=False)

# Create a group for the radiation and set up the values.
nc_file.createDimension('lay', nc_file_rfmip.dimensions['layer'].size)
nc_file.createDimension('lev', nc_file_rfmip.dimensions['level'].size)
nc_file.createDimension('col', nc_file_rfmip.dimensions['site'].size)
nc_file.createDimension('band_lw', band_lw)
nc_file.createDimension('band_sw', band_sw)

nc_pres_layer = nc_file.createVariable('p_lay', float_type, ('lay', 'col'))
nc_pres_level = nc_file.createVariable('p_lev', float_type, ('lev', 'col'))
nc_temp_layer = nc_file.createVariable('t_lay', float_type, ('lay', 'col'))
nc_temp_level = nc_file.createVariable('t_lev', float_type, ('lev', 'col'))

nc_pres_layer[:,:] = nc_file_rfmip.variables['pres_layer'][:,:].transpose()
nc_pres_level[:,:] = nc_file_rfmip.variables['pres_level'][:,:].transpose()
nc_temp_layer[:,:] = (nc_file_rfmip.variables['temp_layer'][expt,:,:]).transpose()
nc_temp_level[:,:] = (nc_file_rfmip.variables['temp_level'][expt,:,:]).transpose()

nc_surface_emissivity = nc_file.createVariable('emis_sfc', float_type, ('col', 'band_lw'))
nc_surface_temperature = nc_file.createVariable('t_sfc', float_type, 'col')

nc_surface_emissivity[:,:] = np.tile( (nc_file_rfmip.variables['surface_emissivity'][:]) [:,None], (1, band_lw) )
nc_surface_temperature[:] = nc_file_rfmip.variables['surface_temperature'][expt,:]

nc_surface_albedo_dir = nc_file.createVariable('sfc_alb_dir', float_type, ('col', 'band_sw'))
nc_surface_albedo_dif = nc_file.createVariable('sfc_alb_dif', float_type, ('col', 'band_sw'))

nc_surface_albedo_dir[:,:] = np.tile( (nc_file_rfmip.variables['surface_albedo'][:]) [:,None], (1, band_sw) )
nc_surface_albedo_dif[:,:] = np.tile( (nc_file_rfmip.variables['surface_albedo'][:]) [:,None], (1, band_sw) )

nc_mu0 = nc_file.createVariable('mu0', float_type, ('col'))
nc_mu0[:] = np.cos(nc_file_rfmip.variables['solar_zenith_angle'][:])

nc_tsi = nc_file.createVariable('tsi', float_type, ('col'))
nc_tsi[:] = nc_file_rfmip.variables['total_solar_irradiance'][:]

nc_h2o = nc_file.createVariable('vmr_h2o', float_type, ('lay', 'col'))
nc_o3  = nc_file.createVariable('vmr_o3' , float_type, ('lay', 'col'))
nc_co2 = nc_file.createVariable('vmr_co2', float_type)
nc_n2o = nc_file.createVariable('vmr_n2o', float_type)
nc_co  = nc_file.createVariable('vmr_co' , float_type)
nc_ch4 = nc_file.createVariable('vmr_ch4', float_type)
nc_o2  = nc_file.createVariable('vmr_o2' , float_type)
nc_n2  = nc_file.createVariable('vmr_n2' , float_type)

# Profiles
nc_h2o[:,:] = nc_file_rfmip.variables['water_vapor'][expt,:,:].transpose() * float(nc_file_rfmip.variables['water_vapor'].units)
nc_o3 [:,:] = nc_file_rfmip.variables['ozone'][expt,:,:].transpose()       * float(nc_file_rfmip.variables['ozone'].units)

# Constants
nc_co2[:] = nc_file_rfmip.variables['carbon_dioxide_GM'][expt]  * float(nc_file_rfmip.variables['carbon_dioxide_GM'].units)
nc_n2o[:] = nc_file_rfmip.variables['nitrous_oxide_GM'][expt]   * float(nc_file_rfmip.variables['nitrous_oxide_GM'].units)
nc_co [:] = nc_file_rfmip.variables['carbon_monoxide_GM'][expt] * float(nc_file_rfmip.variables['carbon_monoxide_GM'].units)
nc_ch4[:] = nc_file_rfmip.variables['methane_GM'][expt]         * float(nc_file_rfmip.variables['methane_GM'].units)
nc_o2 [:] = nc_file_rfmip.variables['oxygen_GM'][expt]          * float(nc_file_rfmip.variables['oxygen_GM'].units)
nc_n2 [:] = nc_file_rfmip.variables['nitrogen_GM'][expt]        * float(nc_file_rfmip.variables['nitrogen_GM'].units)

nc_file_rfmip.close()
nc_file.close()
