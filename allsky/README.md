This is the allsky reference case as contained in the main RTE+RRTMGP repository.
In order to run the test, copy the `test_rte_rrtmgp` executable and the coefficient
files into this directory as `coefficients_lw.nc`, `coefficients_sw.nc`,
`cloud_coefficients_lw.nc`, `cloud_coefficients_sw.nc`.

Follow the steps:

1. `python allsky_init.py`          (preparing the model input data)
2. `./test_rte_rrtmgp`              (run the cases)
3. `./copy_and_rename.sh`           (create the output file with the correct `sw_dn_dir`)
4. `python compare-to_reference.py` (compare output to reference file)
5. `python allsky_plot.py`          (plot the cases in a colormesh per flux)

