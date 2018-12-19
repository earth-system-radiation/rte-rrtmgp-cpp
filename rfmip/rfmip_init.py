import numpy as np
import netCDF4 as nc

# Settings
float_type = "f8"

site = 0
expt = 0

# Save all the input data to NetCDF
nc_file = nc.Dataset("rfmip_input.nc", mode="w", datamodel="NETCDF4", clobber=True)
nc_file_rfmip = nc.Dataset("rfmip.nc", mode="r", datamodel="NETCDF4", clobber=False)

# Create a group for the radiation and set up the values.
nc_group_radiation = nc_file.createGroup("radiation")
nc_group_radiation.createDimension("level", nc_file_rfmip.dimensions["level"].size)
nc_group_radiation.createDimension("layer", nc_file_rfmip.dimensions["layer"].size)
nc_group_radiation.createDimension("col", 1)

nc_pres_level = nc_group_radiation.createVariable("pres_level", float_type, ("level"))
nc_pres_layer = nc_group_radiation.createVariable("pres_layer", float_type, ("layer"))
nc_temp_level = nc_group_radiation.createVariable("temp_level", float_type, ("level"))
nc_temp_layer = nc_group_radiation.createVariable("temp_layer", float_type, ("layer"))

nc_pres_level[:] = nc_file_rfmip.variables["pres_level"][site,:]
nc_pres_layer[:] = nc_file_rfmip.variables["pres_layer"][site,:]
nc_temp_level[:] = nc_file_rfmip.variables["temp_level"][expt,site,:]
nc_temp_layer[:] = nc_file_rfmip.variables["temp_layer"][expt,site,:]

nc_surface_emissivity = nc_group_radiation.createVariable("surface_emissivity", float_type)
nc_surface_temperature = nc_group_radiation.createVariable("surface_temperature", float_type)

nc_surface_emissivity[:] = nc_file_rfmip.variables["surface_emissivity"][site]
nc_surface_temperature[:] = nc_file_rfmip.variables["surface_temperature"][expt,site]

nc_h2o = nc_group_radiation.createVariable("h2o", float_type, ("layer"))
nc_o3  = nc_group_radiation.createVariable("o3" , float_type, ("layer"))
nc_co2 = nc_group_radiation.createVariable("co2", float_type)
nc_n2o = nc_group_radiation.createVariable("n2o", float_type)
nc_co  = nc_group_radiation.createVariable("co" , float_type)
nc_ch4 = nc_group_radiation.createVariable("ch4", float_type)
nc_o2  = nc_group_radiation.createVariable("o2" , float_type)
nc_n2  = nc_group_radiation.createVariable("n2" , float_type)

nc_h2o[:] = nc_file_rfmip.variables["water_vapor"][expt,site,:] * float(nc_file_rfmip.variables["water_vapor"].units)
nc_o3 [:] = nc_file_rfmip.variables["ozone"][expt,site,:]       * float(nc_file_rfmip.variables["ozone"].units)
nc_co2[:] = nc_file_rfmip.variables["carbon_dioxide_GM"][expt]  * float(nc_file_rfmip.variables["carbon_dioxide_GM"].units)
nc_n2o[:] = nc_file_rfmip.variables["nitrous_oxide_GM"][expt]   * float(nc_file_rfmip.variables["nitrous_oxide_GM"].units)
nc_co [:] = nc_file_rfmip.variables["carbon_monoxide_GM"][expt] * float(nc_file_rfmip.variables["carbon_monoxide_GM"].units)
nc_ch4[:] = nc_file_rfmip.variables["methane_GM"][expt]         * float(nc_file_rfmip.variables["methane_GM"].units)
nc_o2 [:] = nc_file_rfmip.variables["oxygen_GM"][expt]          * float(nc_file_rfmip.variables["oxygen_GM"].units)
nc_n2 [:] = nc_file_rfmip.variables["nitrogen_GM"][expt]        * float(nc_file_rfmip.variables["nitrogen_GM"].units)

nc_file_rfmip.close()
nc_file.close()
