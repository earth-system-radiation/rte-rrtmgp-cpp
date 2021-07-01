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
    parser = argparse.ArgumentParser(description='Tuning script for combine_and_reorder_2str_kernel()')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=4)
    parser.add_argument('--block_size_y', type=int, default=4)
    parser.add_argument('--block_size_z', type=int, default=1)
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

    compare_fields(result[-3], tau_ref, 'tau_ref')
    compare_fields(result[-2], ssa_ref, 'ssa_ref')
    compare_fields(result[-1], g_ref, 'g_ref')


# Tuning the kernel
def tune():
    tune_params = dict()
    tune_params['block_size_x'] = [1,2,4,8,16,32,64]
    tune_params['block_size_y'] = [1,2]
    tune_params['block_size_z'] = [1,2,4,8,16,32,64]

    answer = len(args)*[None]
    answer[-3] = tau_ref
    answer[-2] = ssa_ref
    answer[-1] = g_ref

    result, env = kt.tune_kernel(
            kernel_name, kernel_string, problem_size,
            args, tune_params, compiler_options=cp,
            answer=answer, atol=1e-14)

    with open('timings_combine_and_reorder_2str_kernel.json', 'w') as fp:
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
    ngpt = type_int(224)
    tmin = type_float(sys.float_info.min)

    # Kernel input
    tau_abs = np.fromfile('{}/tau_abs.bin'.format(bin_path), dtype=type_float)
    tau_rayleigh = np.fromfile('{}/tau_rayleigh.bin'.format(bin_path), dtype=type_float)

    # Kernel output as reference
    tau_ref = np.fromfile('{}/tau.bin'.format(bin_path), dtype=type_float)
    ssa_ref = np.fromfile('{}/ssa.bin'.format(bin_path), dtype=type_float)
    g_ref = np.fromfile('{}/g.bin'.format(bin_path), dtype=type_float)

    # Kernel output from... kernel
    tau = np.zeros_like(tau_ref)
    ssa = np.zeros_like(ssa_ref)
    g = np.zeros_like(g_ref)

    # KT input
    args = [ncol, nlay, ngpt, tmin,
            tau_abs, tau_rayleigh,
            tau, ssa, g]

    problem_size = (ncol, nlay, ngpt)
    kernel_name = 'combine_and_reorder_2str_kernel<{}>'.format(str_float)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_combine_and_reorder_2str_kernel.json', 'r') as file:
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
