import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

nc_file_run = nc.Dataset('rrtmgp-allsky.nc', 'r')
nc_file_ref = nc.Dataset('ref/rrtmgp-allsky.nc', 'r')

sw_flux_dn_run = nc_file_run.variables['sw_flux_dn'][:,:]
sw_flux_dn_ref = nc_file_ref.variables['sw_flux_dn'][:,:]

sw_flux_dir_run = nc_file_run.variables['sw_flux_dir'][:,:]
sw_flux_dir_ref = nc_file_ref.variables['sw_flux_dir'][:,:]

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
plt.tight_layout()

plt.figure()
plt.subplot(311)
plt.pcolormesh(sw_flux_dir_run)
plt.colorbar()
plt.title('sw_flux_dir')
plt.subplot(312)
plt.pcolormesh(sw_flux_dir_ref)
plt.colorbar()
plt.subplot(313)
plt.pcolormesh(sw_flux_dir_run - sw_flux_dir_ref)
plt.colorbar()
plt.tight_layout()


plt.show()
