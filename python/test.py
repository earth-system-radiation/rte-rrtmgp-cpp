import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import timeit
import radiation


# Read the input data.
nc_file = nc.Dataset('rte_rrtmgp_input.nc', 'r')

vmr_h2o = nc_file.variables['vmr_h2o'][:]
vmr_co2 = nc_file.variables['vmr_co2'][:]
vmr_o3  = nc_file.variables['vmr_o3' ][:]
vmr_n2o = nc_file.variables['vmr_n2o'][:]
# vmr_co  = nc_file.variables['vmr_co' ][:]
vmr_ch4 = nc_file.variables['vmr_ch4'][:]
vmr_o2  = nc_file.variables['vmr_o2' ][:]
vmr_n2  = nc_file.variables['vmr_n2' ][:]

gas_concs = radiation.Gas_concs_wrapper()

# Load the gas concentrations.
gas_concs.set_vmr(b'h2o', vmr_h2o)
gas_concs.set_vmr(b'co2', vmr_co2)
gas_concs.set_vmr(b'o3' , vmr_o3 )
gas_concs.set_vmr(b'n2o', vmr_n2o)
# gas_concs.set_vmr(b'co' , vmr_co )
gas_concs.set_vmr(b'ch4', vmr_ch4)
gas_concs.set_vmr(b'o2' , vmr_o2 )
gas_concs.set_vmr(b'n2' , vmr_n2 )


# Load the thermodynamic variables.
p_lay = nc_file.variables['lay'][:]
p_lev = nc_file.variables['lev'][:]
t_lay = nc_file.variables['t_lay'][:]
t_lev = nc_file.variables['t_lev'][:]

t_sfc = nc_file.variables['t_sfc'][:]
emis_sfc = nc_file.variables['emis_sfc'][:]

nc_file.close()

col_dry = np.zeros((0,0))


# Create the output arrays.
tau = np.zeros((0,0,0))
lay_source = np.zeros((0,0,0))
lev_source_inc = np.zeros((0,0,0))
lev_source_dec = np.zeros((0,0,0))
sfc_source = np.zeros((0,0))

lw_flux_up  = np.zeros(p_lev.shape)
lw_flux_dn  = np.zeros(p_lev.shape)
lw_flux_net = np.zeros(p_lev.shape)
lw_bnd_flux_up  = np.zeros((0,0,0))
lw_bnd_flux_dn  = np.zeros((0,0,0))
lw_bnd_flux_net = np.zeros((0,0,0))


# Initialize the solver.
rad = radiation.Radiation_solver_wrapper(gas_concs, b'coefficients_lw.nc')


# Solve the radiation fluxes.
start = timeit.default_timer()

rad.solve(
        False,
        False,
        gas_concs,
        p_lay, p_lev,
        t_lay, t_lev,
        col_dry,
        t_sfc, emis_sfc,
        tau, lay_source,
        lev_source_inc, lev_source_dec,
        sfc_source,
        lw_flux_up, lw_flux_dn, lw_flux_net,
        lw_bnd_flux_up, lw_bnd_flux_dn, lw_bnd_flux_net)

end = timeit.default_timer()
print('Duration: {} s'.format(end-start))


# Plot some output.
plt.figure()
plt.plot(lw_flux_up[:,0], p_lev[:,0], label='lw_flux_up')
plt.plot(lw_flux_dn[:,0], p_lev[:,0], label='lw_flux_dn')
plt.legend(loc=0, frameon=False)
plt.gca().invert_yaxis()
plt.xlabel(r'flux (W m-2)')
plt.ylabel(r'pressure (Pa)')
plt.tight_layout()
plt.show()
