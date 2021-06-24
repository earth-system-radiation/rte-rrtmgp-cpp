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
    parser = argparse.ArgumentParser(description='Tuning script for compute_tau_rayleigh_kernel()')
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

    result = kt.run_kernel(
            kernel_name, kernel_string, problem_size,
            args, params, compiler_options=cp)

    compare_fields(result[-2], tau_rayleigh_ref, 'tau_rayleigh')


# Tuning the kernel
def tune():
    tune_params = dict()
    tune_params['block_size_x'] = [1,2,4,8,12,14,16,24,32,64]
    tune_params['block_size_y'] = [1,2,4,8,12,14,16,24,32,64]
    tune_params['block_size_z'] = [1,2,4,8,12,14,16,24,32,64]

    answer = len(args)*[None]
    answer[-2] = tau_rayleigh_ref

    result, env = kt.tune_kernel(
            kernel_name, kernel_string, problem_size,
            args, tune_params, compiler_options=cp,
            answer=answer, atol=1e-14)

    with open('timings_compute_tau_rayleigh_kernel.json', 'w') as fp:
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

    ncol = type_int(512)
    nlay = type_int(140)
    nbnd = type_int(14)
    ngpt = type_int(224)
    ngas = type_int(7)
    nflav = type_int(9)
    neta = type_int(9)
    npres = type_int(59)
    ntemp = type_int(14)

    idx_h2o = type_int(1)

    # Kernel input
    gpoint_flavor = np.fromfile('{}/gpoint_flavor_sw.bin'.format(bin_path), dtype=type_int)
    band_lims_gpt = np.fromfile('{}/band_lims_gpt_sw.bin'.format(bin_path), dtype=type_int)
    jeta = np.fromfile('{}/jeta_sw.bin'.format(bin_path), dtype=type_int)
    jtemp = np.fromfile('{}/jtemp_sw.bin'.format(bin_path), dtype=type_int)

    krayl = np.fromfile('{}/krayl_sw.bin'.format(bin_path), dtype=type_float)
    col_gas = np.fromfile('{}/col_gas_sw.bin'.format(bin_path), dtype=type_float)
    col_dry = np.fromfile('{}/col_dry_sw.bin'.format(bin_path), dtype=type_float)
    fminor = np.fromfile('{}/fminor_sw.bin'.format(bin_path), dtype=type_float)

    tropo = np.fromfile('{}/tropo_sw.bin'.format(bin_path), dtype=type_int)

    k = np.zeros(ncol*nlay*ngpt, dtype=type_float)

    # Kernel output as reference
    tau_rayleigh_ref = np.fromfile('{}/tau_rayleigh.bin'.format(bin_path), dtype=type_float)

    # Kernel output from... kernel
    tau_rayleigh = np.zeros_like(tau_rayleigh_ref)

    # KT input
    args = [ncol, nlay, nbnd, ngpt,
            ngas, nflav, neta, npres, ntemp,
            gpoint_flavor,
            band_lims_gpt,
            krayl,
            idx_h2o, col_dry, col_gas,
            fminor, jeta,
            tropo, jtemp,
            tau_rayleigh, k]

    problem_size = (nbnd, nlay, ncol)
    kernel_name = 'compute_tau_rayleigh_kernel<{}>'.format(str_float)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_compute_tau_rayleigh_kernel.json', 'r') as file:
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
