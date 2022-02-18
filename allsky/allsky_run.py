import subprocess
import os
import netCDF4 as nc

def remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

remove('rte_rrtmgp_output.nc')
subprocess.run(['./test_rte_rrtmgp', '--cloud-optics', '--output-bnd-fluxes'])

remove('rrtmgp-allsky.nc')
os.rename('rte_rrtmgp_output.nc', 'rrtmgp-allsky.nc')

nc_file = nc.Dataset('rrtmgp-allsky.nc', 'a')
nc_file.renameDimension('col', 'col_flx')
nc_file.renameVariable('sw_flux_dn_dir', 'sw_flux_dir')
nc_file.close()
