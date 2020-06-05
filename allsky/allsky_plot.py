import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

nc_file_run = nc.Dataset('rrtmgp-allsky.nc', 'r')
nc_file_ref = nc.Dataset('ref/rrtmgp-allsky.nc', 'r')

p_lev = nc_file_run.variables['p_lev'][:,0] / 1e3
cols = np.arange(128)

sw_flux_dn_run = nc_file_run.variables['sw_flux_dn'][:,:]
sw_flux_dn_ref = nc_file_ref.variables['sw_flux_dn'][:,:]

sw_flux_dir_run = nc_file_run.variables['sw_flux_dir'][:,:]
sw_flux_dir_ref = nc_file_ref.variables['sw_flux_dir'][:,:]

plt.figure()
plt.subplot(311)
plt.pcolormesh(cols, p_lev, sw_flux_dn_run)
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('sw_flux_dn')
plt.subplot(312)
plt.pcolormesh(cols, p_lev, sw_flux_dn_ref)
plt.colorbar()
plt.gca().invert_yaxis()
plt.subplot(313)
plt.pcolormesh(cols, p_lev, sw_flux_dn_run - sw_flux_dn_ref)
plt.colorbar()
plt.gca().invert_yaxis()
plt.tight_layout()

plt.figure()
plt.subplot(311)
plt.pcolormesh(cols, p_lev, sw_flux_dir_run)
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('sw_flux_dir')
plt.subplot(312)
plt.pcolormesh(cols, p_lev, sw_flux_dir_ref)
plt.colorbar()
plt.gca().invert_yaxis()
plt.subplot(313)
plt.pcolormesh(cols, p_lev, sw_flux_dir_run - sw_flux_dir_ref)
plt.colorbar()
plt.gca().invert_yaxis()
plt.tight_layout()

plt.figure()
plt.plot(sw_flux_dn_run[:,2], p_lev, 'C0-', label='clear')
plt.plot(sw_flux_dn_ref[:,2], p_lev, 'C0:')
plt.plot(sw_flux_dn_run[:,0], p_lev, 'C1-', label='cloud')
plt.plot(sw_flux_dn_ref[:,0], p_lev, 'C1:')
plt.gca().invert_yaxis()
plt.title('sw_flux_dn')
plt.legend(loc=0, frameon=False)
plt.tight_layout()

plt.show()
