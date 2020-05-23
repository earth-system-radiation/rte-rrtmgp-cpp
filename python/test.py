import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import radiation


# Read the input data.
nc_file = nc.Dataset('rte_rrtmgp_input.nc', 'r')

vmr_h2o = nc_file.variables['vmr_h2o'][:]
vmr_co2 = nc_file.variables['vmr_co2'][:]
vmr_o3  = nc_file.variables['vmr_o3' ][:]
rmr_n2o = nc_file.variables['vmr_n2o'][:]
# vmr_co  = nc_file.variables['vmr_co' ][:]
vmr_ch4 = nc_file.variables['vmr_ch4'][:]
vmr_o2  = nc_file.variables['vmr_o2' ][:]
vmr_n2  = nc_file.variables['vmr_n2' ][:]

nc_file.close()


rad = radiation.Radiation_solver_wrapper()

rad.set_vmr(b'h2o', vmr_h2o)
rad.set_vmr(b'co2', vmr_co2)
rad.set_vmr(b'o3' , vmr_o3 )
rad.set_vmr(b'n2o', rmr_n2o)
# rad.set_vmr(b'co' , vmr_co )
rad.set_vmr(b'ch4', vmr_ch4)
rad.set_vmr(b'o2' , vmr_o2 )
rad.set_vmr(b'n2' , vmr_n2 )

rad.load_kdistribution_lw()

