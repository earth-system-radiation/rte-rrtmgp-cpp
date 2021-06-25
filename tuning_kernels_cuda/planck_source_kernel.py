import kernel_tuner as kt
import numpy as np
import argparse
import json
import os

# Path where data is stored
bin_path = '../rcemip'


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for Planck_source_kernel()')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=14)
    parser.add_argument('--block_size_y', type=int, default=1)
    parser.add_argument('--block_size_z', type=int, default=32)
    return parser.parse_args()


# Compare results against reference
def compare_fields(arr1, arr2, name):
    okay = np.allclose(arr1, arr2, atol=1e-15)
    max_diff = np.abs(arr1-arr2).max()
    if okay:
        print('results for {}: OKAY!'.format(name))
    else:
        print('results for {}: NOT OKAY, max diff={}'.format(name, max_diff))


# Run one instance of the kernel and test output
def run_and_test(params: dict):

    print('Running {} [block_size_x: {}, block_size_y: {}, block_size_z: {}]'.format(
            kernel_name,
            params['block_size_x'],
            params['block_size_y'],
            params['block_size_z']))

    params['RTE_RRTMGP_USE_CBOOL'] = 1

    result = kt.run_kernel(
            kernel_name, kernel_string, problem_size,
            args, params, compiler_options=cp)

    compare_fields(result[-6], sfc_src_ref, 'sfc_src')
    compare_fields(result[-5], lay_src_ref, 'lay_src')
    compare_fields(result[-4], lev_src_inc_ref, 'lev_src_inc')
    compare_fields(result[-3], lev_src_dec_ref, 'lev_srd_dec')
    compare_fields(result[-2], sfc_src_jac_ref, 'sfc_src_jac')


# Tuning the kernel
def tune():
    tune_params = dict()
    tune_params['RTE_RRTMGP_USE_CBOOL'] = 1
    tune_params['block_size_x'] = [1, 2, 3, 4, 5, 6, 7, 8, 14]
    tune_params['block_size_y'] = [1, 2, 3, 4]
    tune_params['block_size_z'] = [1, 2, 3, 4, 32]

    answer = len(args)*[None]
    answer[-6] = sfc_src_ref
    answer[-5] = lay_src_ref
    answer[-4] = lev_src_inc_ref
    answer[-3] = lev_src_dec_ref
    answer[-2] = sfc_src_jac_ref

    result, env = kt.tune_kernel(
            kernel_name, kernel_string, problem_size,
            args, tune_params, compiler_options=cp,
            answer=answer, atol=1e-14)

    with open('timings_planck_source_kernel.json', 'w') as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    command_line = parse_command_line()

    # CUDA source code
    with open('../src_kernels_cuda/gas_optics_kernels.cu') as f:
        kernel_string = f.read()

    # Settings
    type_int = np.int32
    type_float = np.float64
    type_bool = np.ubyte  # = default without `RTE_RRTMGP_USE_CBOOL`

    str_float = 'float' if type_float is np.float32 else 'double'
    include = os.path.abspath('../include')
    cp = ['-I{}'.format(include)]

    ncol = type_int(512)
    nlay = type_int(140)
    nband = type_int(16)
    ngpt = type_int(256)
    nflav = type_int(10)
    neta = type_int(9)
    npres = type_int(59)
    ntemp = type_int(14)
    ngas = type_int(7)

    nplancktemp = type_int(196)
    sfc_lay = type_int(1)

    temp_ref_min = type_float(160)
    totplnk_delta = type_float(1)
    delta_Tsurf = type_float(1)

    # Kernel input
    tlay = np.fromfile('{}/tlay.bin'.format(bin_path), dtype=type_float)
    tlev = np.fromfile('{}/tlev.bin'.format(bin_path), dtype=type_float)
    tsfc = np.fromfile('{}/tsfc.bin'.format(bin_path), dtype=type_float)
    fmajor = np.fromfile('{}/fmajor.bin'.format(bin_path), dtype=type_float)
    pfracin = np.fromfile('{}/pfracin.bin'.format(bin_path), dtype=type_float)
    totplnk = np.fromfile('{}/totplnk.bin'.format(bin_path), dtype=type_float)

    jeta = np.fromfile('{}/jeta.bin'.format(bin_path), dtype=type_int)
    jtemp = np.fromfile('{}/jtemp.bin'.format(bin_path), dtype=type_int)
    jpress = np.fromfile('{}/jpress.bin'.format(bin_path), dtype=type_int)
    gpoint_bands = np.fromfile('{}/gpoint_bands.bin'.format(bin_path), dtype=type_int)
    gpoint_flavor = np.fromfile('{}/gpoint_flavor.bin'.format(bin_path), dtype=type_int)
    band_lims_gpt = np.fromfile('{}/band_lims_gpt.bin'.format(bin_path), dtype=type_int)

    tropo = np.fromfile('{}/tropo.bin'.format(bin_path), dtype=type_bool)

    # Reference kernel output
    sfc_src_ref = np.fromfile('{}/sfc_src.bin'.format(bin_path), dtype=type_float)
    lay_src_ref = np.fromfile('{}/lay_src.bin'.format(bin_path), dtype=type_float)
    lev_src_inc_ref = np.fromfile('{}/lev_src_inc.bin'.format(bin_path), dtype=type_float)
    lev_src_dec_ref = np.fromfile('{}/lev_src_dec.bin'.format(bin_path), dtype=type_float)
    sfc_src_jac_ref = np.fromfile('{}/sfc_src_jac.bin'.format(bin_path), dtype=type_float)

    # Kernel tuner kernel output
    sfc_src = np.zeros_like(sfc_src_ref, dtype=type_float)
    lay_src = np.zeros_like(lay_src_ref, dtype=type_float)
    lev_src_inc = np.zeros_like(lev_src_inc_ref, dtype=type_float)
    lev_src_dec = np.zeros_like(lev_src_dec_ref, dtype=type_float)
    sfc_src_jac = np.zeros_like(sfc_src_jac_ref, dtype=type_float)

    # Misc
    ones = np.ones(2, dtype=type_float)
    pfrac = np.zeros_like(lay_src, dtype=type_float)

    args = [
        ncol, nlay, nband, ngpt,
        nflav, neta, npres, ntemp, nplancktemp,
        tlay, tlev, tsfc, sfc_lay,
        fmajor, jeta, tropo, jtemp,
        jpress, gpoint_bands, band_lims_gpt,
        pfracin, temp_ref_min, totplnk_delta,
        totplnk, gpoint_flavor, ones,
        delta_Tsurf, sfc_src, lay_src,
        lev_src_inc, lev_src_dec,
        sfc_src_jac, pfrac]

    problem_size = (nband, nlay, ncol)
    kernel_name = 'Planck_source_kernel<{}>'.format(str_float)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_planck_source_kernel.json', 'r') as file:
                configurations = json.load(file)
            best_configuration = min(configurations, key=lambda x: x['time'])
            parameters['block_size_x'] = best_configuration['block_size_x']
            parameters['block_size_y'] = best_configuration['block_size_y']
            parameters['block_size_z'] = best_configuration['block_size_z']
        else:
            parameters['block_size_x'] = command_line.block_size_x
            parameters['block_size_y'] = command_line.block_size_y
            parameters['block_size_z'] = command_line.block_size_z
        run_and_test(parameters)
