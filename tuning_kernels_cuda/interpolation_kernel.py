import kernel_tuner as kt
import numpy as np
import argparse
import json
import sys
import os

# Path where data is stored
bin_path = '../rcemip'

# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for interpolation_kernel()')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=16)
    parser.add_argument('--block_size_y', type=int, default=32)
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
    print('Running {} [block_size_x: {}, block_size_y: {}]'.format(
            kernel_name,
            params['block_size_x'],
            params['block_size_y']))

    result = kt.run_kernel(
            kernel_name, kernel_string, problem_size,
            args, params, compiler_options=cp)

    compare_fields(result[-6], fmajor_ref, 'fmajor')
    compare_fields(result[-5], fminor_ref, 'fminor')
    compare_fields(result[-4], col_mix_ref, 'col_mix')
    compare_fields(result[-3], tropo_ref, 'tropo')
    compare_fields(result[-2], jeta_ref, 'jeta')
    compare_fields(result[-1], jpress_ref, 'jpress')

# Tuning the kernel
def tune():
    tune_params = dict()
    tune_params['block_size_x'] = [1,2,4,8,12,16,24,32]
    tune_params['block_size_y'] = [1,2,4,8,12,16,24,32]

    answer = len(args)*[None]
    answer[-6] = fmajor_ref
    answer[-5] = fminor_ref
    answer[-4] = col_mix_ref
    answer[-3] = tropo_ref
    answer[-2] = jeta_ref
    answer[-1] = jpress_ref

    result, env = kt.tune_kernel(
            kernel_name, kernel_string, problem_size,
            args, tune_params, compiler_options=cp,
            answer=answer, atol=1e-14)

    with open('timings_interpolation_kernel.json', 'w') as fp:
        json.dump(result, fp)

if __name__ == '__main__':
    command_line = parse_command_line()

    # CUDA source code
    with open('../src_kernels_cuda/gas_optics_kernels.cu') as f:
        kernel_string = f.read()

    # Settings
    type_int = np.int32
    type_float = np.float64
    type_bool = np.int32  # = default without `RTE_RRTMGP_USE_CBOOL`

    str_float = 'float' if type_float is np.float32 else 'double'
    include = os.path.abspath('../include')
    cp = ['-I{}'.format(include)]

    ncol = type_int(144)
    nlay = type_int(140)
    ngas = type_int(7)
    nflav = type_int(10)
    neta = type_int(9)
    npres = type_int(59)
    ntemp = type_int(14)

    tmin = type_float(sys.float_info.min)
    press_ref_log_delta = type_float(-0.20000000000000617)
    temp_ref_min = type_float(160)
    temp_ref_delta = type_float(15)
    press_ref_trop_log = type_float(9.205170185988093)

    # Kernel input
    flavor = np.fromfile('{}/flavor.bin'.format(bin_path), dtype=type_int)

    press_ref_log = np.fromfile('{}/press_ref_log.bin'.format(bin_path), dtype=type_float)
    temp_ref = np.fromfile('{}/temp_ref.bin'.format(bin_path), dtype=type_float)
    vmr_ref = np.fromfile('{}/vmr_ref.bin'.format(bin_path), dtype=type_float)
    play = np.fromfile('{}/play.bin'.format(bin_path), dtype=type_float)
    tlay = np.fromfile('{}/tlay.bin'.format(bin_path), dtype=type_float)
    col_gas = np.fromfile('{}/col_gas.bin'.format(bin_path), dtype=type_float)

    # Kernel output as reference.
    jtemp_ref = np.fromfile('{}/jtemp.bin'.format(bin_path), dtype=type_int)
    jpress_ref = np.fromfile('{}/jpress.bin'.format(bin_path), dtype=type_int)
    jeta_ref = np.fromfile('{}/jeta.bin'.format(bin_path), dtype=type_int)

    tropo_ref = np.fromfile('{}/tropo.bin'.format(bin_path), dtype=type_bool)

    col_mix_ref = np.fromfile('{}/col_mix.bin'.format(bin_path), dtype=type_float)
    fminor_ref = np.fromfile('{}/fminor.bin'.format(bin_path), dtype=type_float)
    fmajor_ref = np.fromfile('{}/fmajor.bin'.format(bin_path), dtype=type_float)

    # Kernel output from... kernel
    jtemp = np.zeros_like(jtemp_ref)
    jpress = np.zeros_like(jpress_ref)
    jeta = np.zeros_like(jeta_ref)

    tropo = np.zeros_like(tropo_ref)

    col_mix = np.zeros_like(col_mix_ref)
    fminor = np.zeros_like(fminor_ref)
    fmajor = np.zeros_like(fmajor_ref)

    # KT input
    args = [ncol, nlay, ngas, nflav,
            neta, npres, ntemp, tmin,
            flavor, press_ref_log, temp_ref,
            press_ref_log_delta, temp_ref_min,
            temp_ref_delta, press_ref_trop_log,
            vmr_ref, play, tlay, col_gas,
            jtemp, fmajor, fminor, col_mix,
            tropo, jeta, jpress]

    problem_size = (ncol, nlay)
    kernel_name = 'interpolation_kernel<{}>'.format(str_float)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_interpolation_kernel.json', 'r') as file:
                configurations = json.load(file)
            best_configuration = min(configurations, key=lambda x: x['time'])
            parameters['block_size_x'] = best_configuration['block_size_x']
            parameters['block_size_y'] = best_configuration['block_size_y']
        else:
            parameters['block_size_x'] = command_line.block_size_x
            parameters['block_size_y'] = command_line.block_size_y

        run_and_test(parameters)
