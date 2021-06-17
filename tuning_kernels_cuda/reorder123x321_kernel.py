import kernel_tuner as kt
import numpy as np
import argparse
import json
import os


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='reorder123x321_kernel()') 
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=16)
    parser.add_argument('--block_size_y', type=int, default=1)
    parser.add_argument('--block_size_z', type=int, default=16)
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
            kernel_name, params['block_size_x'], params['block_size_y'], params['block_size_z']))

    result = kt.run_kernel(
            kernel_name, kernel_string, problem_size,
            args, params, compiler_options=cp,
            smem_args={"size": 1024 * type_float.itemsize})

    compare_fields(result[-1], ref, '123x321')


# Tuning the kernel
def tune():
    tune_params = dict()
    tune_params['block_size_x'] = [2**i for i in range(0, 11)]
    tune_params['block_size_y'] = [2**i for i in range(0, 11)]
    tune_params['block_size_z'] = [2**i for i in range(0, 11)]

    result, env = kt.tune_kernel(
            kernel_name, kernel_string, problem_size,
            args, tune_params, compiler_options=cp,
            smem_args={"size": 1024 * type_float.itemsize})

    with open('reorder123x321_kernel.json', 'w') as fp:
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

    ni = type_int(144)
    nj = type_int(140)
    nk = type_int(256)
    n = ni * nj * nk

    # Kernel input and output
    arr_in  = np.random.random(n).astype(type_float)
    arr_out = np.zeros(n, dtype=type_float)

    # Reference transposed field
    ref = arr_in.reshape((ni,nj,nk))
    ref = ref.transpose((2,1,0)).flatten()

    # Kernel tuner
    args = [ni, nj, nk, arr_in, arr_out]

    problem_size = (ni, nj, nk)
    kernel_name = 'reorder123x321_kernel<{}>'.format(str_float)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_reorder123x321_kernel.json', 'r') as file:
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
