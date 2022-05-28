import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as pl

### input ###
azimuth_angle = 235.
zenith_angle = 45
albedo = 0.2
tod_flux_direct = 95
tod_flux_diffuse = 5

# optical properties
tau_clear = 0.1
tau_cloud = 10.
ssa_clear = 0.5
ssa_cloud = 0.9
asy_param = .86

#grid settings
nx = 256
ny = 256
nz = 128

dx = 25
dy = 25
dz = 25

# rectangular cloud(s) properties
cld_bot_idx = 64
cld_top_idx = 96
cloud_size_x = 32
cloud_size_y = 64
n_cloud_x = 4
n_cloud_y = 2

### create the rte_rrtmgp_input file
# set cloud mask
cld_mask = np.zeros((nz,ny,nx))
void_x = int((nx - (cloud_size_x * n_cloud_x)) / n_cloud_x)
void_y = int((ny - (cloud_size_y * n_cloud_y)) / n_cloud_y)

for j in range(n_cloud_y):
    for i in range(n_cloud_x):
        i0 = i*(cloud_size_x+void_x)
        j0 = j*(cloud_size_y+void_y)
        cld_mask[cld_bot_idx:cld_top_idx, j0:j0+cloud_size_y, i0:i0+cloud_size_x] = 1

ncf = nc.Dataset("rt_input.nc", "w")
ncf.createDimension("x", nx)
ncf.createDimension("y", ny)
ncf.createDimension("z", nz)
nc_x = ncf.createVariable("x", "f8", ("x",))
nc_y = ncf.createVariable("y", "f8", ("y",))
nc_z = ncf.createVariable("z", "f8", ("z",))
nc_x[:] = np.arange(nx) * dx
nc_y[:] = np.arange(ny) * dy
nc_z[:] = np.arange(nz) * dz

tau_cld = cld_mask * (tau_cloud / (cld_top_idx - cld_bot_idx))
tau_tot = np.ones((nz, ny, nx)) * (tau_clear / 128) + tau_cld
ssa = (tau_cld*ssa_cloud + tau_tot * ssa_clear)/(tau_cld + tau_tot)
asy = cld_mask * asy_param

nc_tau_tot = ncf.createVariable("tau_tot", "f8", ("z", "y", "x"))
nc_tau_cld = ncf.createVariable("tau_cld", "f8", ("z", "y", "x"))
nc_ssa = ncf.createVariable("ssa", "f8", ("z", "y", "x"))
nc_asy = ncf.createVariable("asy", "f8", ("z", "y", "x"))

nc_tau_tot[:] = tau_tot
nc_tau_cld[:] = tau_cld
nc_ssa[:] = ssa
nc_asy[:] = asy

nc_azi = ncf.createVariable("azi", "f8")
nc_sza = ncf.createVariable("sza", "f8")
nc_alb = ncf.createVariable("albedo", "f8")
nc_dir = ncf.createVariable("tod_direct", "f8")
nc_dif = ncf.createVariable("tod_diffuse", "f8")

nc_azi[:] = np.deg2rad(azimuth_angle)
nc_sza[:] = np.deg2rad(zenith_angle)
nc_alb[:] = albedo
nc_dir[:] = tod_flux_direct
nc_dif[:] = tod_flux_diffuse

ncf.close()
