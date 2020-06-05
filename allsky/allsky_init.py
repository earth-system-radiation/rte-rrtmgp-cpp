import numpy as np
import netCDF4 as nc

# Settings
float_type = 'f8'

band_lw = 16
band_sw = 14

# Save all the input data to NetCDF
nc_file = nc.Dataset('rte_rrtmgp_input.nc', mode='w', datamodel='NETCDF4', clobber=True)
nc_file_garand = nc.Dataset('garand-atmos-1.nc', mode='r', datamodel='NETCDF4', clobber=False)

# Create a group for the radiation and set up the values.
nc_file.createDimension('lay', nc_file_garand.dimensions['lay'].size)
nc_file.createDimension('lev', nc_file_garand.dimensions['lev'].size)
nc_file.createDimension('col', nc_file_garand.dimensions['col'].size)
nc_file.createDimension('band_lw', band_lw)
nc_file.createDimension('band_sw', band_sw)

nc_pres_layer = nc_file.createVariable('p_lay', float_type, ('lay', 'col'))
nc_pres_level = nc_file.createVariable('p_lev', float_type, ('lev', 'col'))
nc_temp_layer = nc_file.createVariable('t_lay', float_type, ('lay', 'col'))
nc_temp_level = nc_file.createVariable('t_lev', float_type, ('lev', 'col'))

nc_pres_layer[:,:] = nc_file_garand.variables['p_lay'][:,:]
nc_pres_level[:,:] = nc_file_garand.variables['p_lev'][:,:]

# Make sure the top edge does not exceed the minimum tolerable pressure
# of the coefficient files.
nc_pres_level[:,:] = np.maximum(nc_pres_level[:,:], np.nextafter(1.005183574463, 1.))

nc_temp_layer[:,:] = nc_file_garand.variables['t_lay'][:,:]
nc_temp_level[:,:] = nc_file_garand.variables['t_lev'][:,:]

nc_col_dry = nc_file.createVariable('col_dry', float_type, ('lay', 'col'))
nc_col_dry[:,:] = nc_file_garand.variables['col_dry'][:,:]

nc_surface_emissivity = nc_file.createVariable('emis_sfc', float_type, ('col', 'band_lw'))
nc_surface_temperature = nc_file.createVariable('t_sfc', float_type, 'col')

nc_surface_emissivity[:,:] = 0.98
nc_surface_temperature[:] = nc_temp_level[0,:]

nc_surface_albedo_dir = nc_file.createVariable('sfc_alb_dir', float_type, ('col', 'band_sw'))
nc_surface_albedo_dif = nc_file.createVariable('sfc_alb_dif', float_type, ('col', 'band_sw'))

nc_surface_albedo_dir[:,:] = 0.06
nc_surface_albedo_dif[:,:] = 0.06

nc_mu0 = nc_file.createVariable('mu0', float_type, ('col'))
nc_mu0[:] = 0.86

nc_h2o     = nc_file.createVariable('vmr_h2o', float_type, ('lay', 'col'))
nc_co2     = nc_file.createVariable('vmr_co2', float_type, ('lay', 'col'))
nc_o3      = nc_file.createVariable('vmr_o3' , float_type, ('lay', 'col'))
nc_n2o     = nc_file.createVariable('vmr_n2o', float_type, ('lay', 'col'))
nc_co      = nc_file.createVariable('vmr_co' , float_type, ('lay', 'col'))
nc_ch4     = nc_file.createVariable('vmr_ch4', float_type, ('lay', 'col'))
nc_o2      = nc_file.createVariable('vmr_o2' , float_type, ('lay', 'col'))
nc_n2      = nc_file.createVariable('vmr_n2' , float_type, ('lay', 'col'))

# Profiles
nc_h2o[:,:] = nc_file_garand.variables['vmr_h2o'][:,:]
nc_co2[:,:] = nc_file_garand.variables['vmr_co2'][:,:]
nc_o3 [:,:] = nc_file_garand.variables['vmr_o3' ][:,:]
nc_n2o[:,:] = nc_file_garand.variables['vmr_n2o'][:,:]
nc_co [:,:] = nc_file_garand.variables['vmr_co' ][:,:]
nc_ch4[:,:] = nc_file_garand.variables['vmr_ch4'][:,:]
nc_o2 [:,:] = nc_file_garand.variables['vmr_o2' ][:,:]
nc_n2 [:,:] = nc_file_garand.variables['vmr_n2' ][:,:]

nc_file_garand.close()
nc_file.close()
