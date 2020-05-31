This is the RFMIP reference case as contained in the main RTE+RRTMGP repository.
The case can be run using the following steps:

`python stage_files.py`          (downloading the reference data)
`python rfmip_init.py`           (preparing the model input data)
`python rfmip_run.py`            (run the 1800 cases)
`python rfmip_plot.py`           (plot the cases in a colormesh per flux)
`python compare-to_reference.py` (compare to reference file)

