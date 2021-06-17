import kernel_tuner as kt
import numpy as np
import argparse
import json
import sys
import os


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='sw_2stream_kernel()') 
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=16)
    parser.add_argument('--block_size_y', type=int, default=16)
    parser.add_argument('--block_size_z', type=int, default=1)
    return parser.parse_args()


# Run one instance of the kernel and test output
def run_and_test(params: dict):
    print('Running {} [block_size_x: {}, block_size_y: {}]'.format(
            kernel_name, params['block_size_x'], params['block_size_y']))

    result = kt.run_kernel(
            kernel_name, kernel_string, problem_size,
            args, params, compiler_options=cp)


# Tuning the kernel
def tune():
    tune_params = dict()
    tune_params['block_size_x'] = [1,2,4,8,12,16,24,32,64]
    tune_params['block_size_y'] = [1,2,4,8,12,16,24,32,64]
    tune_params['block_size_z'] = [1,2,4,8,12,16,24,32,64]

    result, env = kt.tune_kernel(
            kernel_name, kernel_string, problem_size,
            args, tune_params, compiler_options=cp)

    with open('timings_sw_2stream_kernel.json', 'w') as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    command_line = parse_command_line()

    # CUDA source code
    with open('../src_kernels_cuda/rte_solver_kernels.cu') as f:
        kernel_string = f.read()

    # Settings
    type_int = np.int32
    type_float = np.float64
    type_bool = np.int32  # = default without `RTE_RRTMGP_USE_CBOOL`

    str_float = 'float' if type_float is np.float32 else 'double'
    include = os.path.abspath('../include')
    cp = ['-I{}'.format(include)]

    ncol = type_int(128)
    nlay = type_int(140)
    ngpt = type_int(224)

    opt_size = ncol * nlay * ngpt
    flx_size = ncol * (nlay+1) * ngpt
    alb_size = ncol * ngpt

    # Input arrays; for this kernel, the values don't matter..
    tau  = np.zeros(opt_size, dtype=type_float)
    ssa  = np.zeros(opt_size, dtype=type_float)
    g = np.zeros(opt_size, dtype=type_float)
    mu0 = np.zeros(ncol, dtype=type_float)

    # Output arrays
    r_dif = np.zeros(opt_size, dtype=type_float);
    t_dif = np.zeros(opt_size, dtype=type_float);
    r_dir = np.zeros(opt_size, dtype=type_float);
    t_dir = np.zeros(opt_size, dtype=type_float);
    t_noscat = np.zeros(opt_size, dtype=type_float);

    tmin = type_float(sys.float_info.min)

    args = [ncol, nlay, ngpt, tmin,
            tau, ssa,
            g, mu0,
            r_dif, t_dif,
            r_dir, t_dir,
            t_noscat]

    problem_size = (ncol, nlay, ngpt)
    kernel_name = 'sw_2stream_kernel<{}>'.format(str_float)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_sw_2stream_kernel.json', 'r') as file:
                configurations = json.load(file)
            best_configuration = min(configurations, key=lambda x: x['time'])
            parameters['block_size_x'] = best_configuration['block_size_x']
            parameters['block_size_y'] = best_configuration['block_size_y']
        else:
            parameters['block_size_x'] = command_line.block_size_x
            parameters['block_size_y'] = command_line.block_size_y
        run_and_test(parameters)
