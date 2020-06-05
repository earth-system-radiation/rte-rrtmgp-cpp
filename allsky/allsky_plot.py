import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

nc_file_run = nc.Dataset('rrtmgp-allsky.nc', 'r')
nc_file_ref = nc.Dataset('ref/rrtmgp-allsky.nc', 'r')

sw_flux_dn_run = nc_file_run.variables['sw_flux_dn'][:,:]
sw_flux_dn_ref = nc_file_ref.variables['sw_flux_dn'][:,:2]

plt.figure()
plt.subplot(311)
plt.pcolormesh(sw_flux_dn_run)
plt.colorbar()
plt.title('sw_flux_dn')
plt.subplot(312)
plt.pcolormesh(sw_flux_dn_ref)
plt.colorbar()
plt.subplot(313)
plt.pcolormesh(sw_flux_dn_run - sw_flux_dn_ref)
plt.colorbar()
plt.show()
