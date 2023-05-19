#! /bin/sh
ln -sf ../rte-rrtmgp/rrtmgp-data/rrtmgp-cloud-optics-coeffs-reordered-sw.nc cloud_coefficients_sw.nc
ln -sf ../rte-rrtmgp/rrtmgp-data/rrtmgp-cloud-optics-coeffs-lw.nc cloud_coefficients_lw.nc
ln -sf ../rte-rrtmgp/rrtmgp-data/rrtmgp-data-sw-g224-2018-12-04.nc coefficients_sw.nc
ln -sf ../rte-rrtmgp/rrtmgp-data/rrtmgp-data-lw-g256-2018-12-04.nc coefficients_lw.nc
ln -sf ../build/test_rte_rrtmgp .
