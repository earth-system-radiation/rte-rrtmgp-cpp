#!/usr/bin/env python
import sys
import kernel_tuner as kt
import numpy as np
import argparse
import json
import common

from collections import OrderedDict


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='sw_source_adding_kernel()')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=32)
    parser.add_argument('--block_size_y', type=int, default=4)
    return parser.parse_args()


# Run one instance of the kernel and test output

def run_and_test(params: dict):

    #call first reference kernel
    two_stream_result = kt.run_kernel(
            two_stream_ref_kernel_name, ref_kernels_src, two_stream_ref_problem_size,
            two_stream_args, dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=64, block_size_y=1, block_size_z=1),
            compiler_options=common.cp)

    #put the input from the 2stream reference kernel into the input of the source reference kernel
    source_kernel_args[4] = two_stream_result[-3]
    source_kernel_args[5] = two_stream_result[-2]
    source_kernel_args[6] = two_stream_result[-1]

    source_ref_result = kt.run_kernel(
            source_ref_kernel_name, ref_kernels_src, ref_problem_size,
            source_kernel_args, dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=64, block_size_y=11),
            compiler_options=common.cp)

    #call the fused kernel
    fused_result = kt.run_kernel(
            fused_kernel_name, kernels_src, ref_problem_size,
            fused_kernel_args, dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=16, block_size_y=16),
            compiler_options=common.cp)

    #test source outputs
    outputs = ["flux_dir", "source_sfc", "source_dn", "source_up"]
    for i, o in enumerate(outputs):
        okay = np.allclose(fused_result[-1-i], source_ref_result[-1-i], atol=1e-14)
        max_diff = np.abs(fused_result[-1-i], source_ref_result[-1-i]).max()
        if okay:
            print(f'results for {o}: OKAY!')
        else:
            print(f'results for {o}: NOT OKAY, {max_diff}')
            print("Result:")
            print(fused_result[-1-i])
            print("Expected:")
            print(source_ref_result[-1-i])

    #test 2stream outputs r_dif t_dif
    outputs = ["t_dif", "r_dif"]
    for i, o in enumerate(outputs):
        okay = np.allclose(fused_result[-6-i], two_stream_result[-4-i], atol=1e-14)
        max_diff = np.abs(fused_result[-6-i], two_stream_result[-4-i]).max()
        if okay:
            print(f'results for {o}: OKAY!')
        else:
            print(f'results for {o}: NOT OKAY, {max_diff}')
            print("Result:")
            print(fused_result[-6-i])
            print("Expected:")
            print(two_stream_result[-4-i])



def tune_fused_kernel():

    tune_params = OrderedDict()
    tune_params['RTE_RRTMGP_USE_CBOOL'] = [1]
    tune_params['block_size_x'] = [16*i for i in range(1,33)]
    tune_params['block_size_y'] = [1, 2, 3, 4, 5, 6, 7, 8,9, 10, 11, 12, 16]
    #unrolling the nlay loop was not beneficial in most configurations
    #tune_params['loop_unroll_factor_nlay'] = [0]+[i for i in range(1, nlay+1) if nlay//i == nlay/i]

    #call first reference kernel
    two_stream_result = kt.run_kernel(
            two_stream_ref_kernel_name, ref_kernels_src, two_stream_ref_problem_size,
            two_stream_args, dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=64, block_size_y=1, block_size_z=1),
            compiler_options=common.cp)

    #put the input from the 2stream reference kernel into the input of the source reference kernel
    source_kernel_args[4] = two_stream_result[-3]
    source_kernel_args[5] = two_stream_result[-2]
    source_kernel_args[6] = two_stream_result[-1]

    source_ref_result = kt.run_kernel(
            source_ref_kernel_name, ref_kernels_src, ref_problem_size,
            source_kernel_args, dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=64, block_size_y=11),
            compiler_options=common.cp)

    answer = [None for _ in fused_kernel_args]
    answer[-1] = source_ref_result[-1] #flux_dir
    answer[-2] = source_ref_result[-2] #source_sfc
    answer[-3] = source_ref_result[-3] #source_dn
    answer[-4] = source_ref_result[-4] #source_up

    answer[-6] = two_stream_result[-4] #t_dif
    answer[-7] = two_stream_result[-5] #r_dif

    #tune the fused kernel
    fused_result = kt.tune_kernel(
            fused_kernel_name, kernels_src, ref_problem_size,
            fused_kernel_args, tune_params, compiler_options=common.cp,
            answer=answer, atol=1e-14,
            verbose=True, observers=[common.reg_observer], metrics=common.metrics, iterations=32)

    with open('timings_sw_source_2stream_kernel.json', 'w') as fp:
        json.dump(fused_result, fp)




def tune_two_stream_ref():

    tune_params = OrderedDict()
    tune_params['RTE_RRTMGP_USE_CBOOL'] = [1]
    tune_params['block_size_x'] = [16, 32, 64, 128, 256]
    tune_params['block_size_y'] = [1, 4, 8, 16]
    tune_params['block_size_z'] = [1, 4, 8]

    #ref 2stream_kernel is 883us on A100
    kt.tune_kernel(
            two_stream_ref_kernel_name, ref_kernels_src, two_stream_ref_problem_size,
            two_stream_args, tune_params, compiler_options=common.cp)

def tune_source_ref():

    tune_params = OrderedDict()
    tune_params['RTE_RRTMGP_USE_CBOOL'] = [1]
    tune_params['block_size_x'] = [16, 32, 64, 128, 256]
    tune_params['block_size_y'] = [1, 4, 8, 16]

    #ref kernel is about 662us on A100
    kt.tune_kernel(
            source_ref_kernel_name, ref_kernels_src, ref_problem_size,
            source_kernel_args, tune_params, compiler_options=common.cp)




if __name__ == '__main__':
    command_line = parse_command_line()

    kernels_src = common.dir_name + '../src_kernels_cuda/rte_solver_kernels.cu'
    ref_kernels_src = common.dir_name + 'reference_kernels/rte_solver_kernels.cu'

    # Settings
    type_bool = np.int8

    ncol = common.type_int(512)
    nlay = common.type_int(140)
    ngpt = common.type_int(224)

    top_at_1 = type_bool(1)

    opt_size = ncol * nlay * ngpt
    flx_size = ncol * (nlay+1) * ngpt
    alb_size = ncol * ngpt

    tau  = np.random.random(opt_size).astype(common.type_float)
    ssa  = np.random.random(opt_size).astype(common.type_float)
    g = np.random.random(opt_size).astype(common.type_float)
    mu0 = np.random.random(ncol).astype(common.type_float)

    tmin = common.type_float(sys.float_info.min)

    # Input arrays; for this kernel, the values don't matter..
    t_noscat = np.random.random(opt_size).astype(common.type_float)
    flux_up  = np.random.random(flx_size).astype(common.type_float)
    flux_dn  = np.random.random(flx_size).astype(common.type_float)
    flux_dir = 1+np.random.random(flx_size).astype(common.type_float)

    sfc_alb_dir = np.random.random(alb_size).astype(common.type_float)
    sfc_alb_dif = np.random.random(alb_size).astype(common.type_float)

    # Output arrays
    r_dif = np.zeros(opt_size, dtype=common.type_float)
    t_dif = np.zeros(opt_size, dtype=common.type_float)
    r_dir = np.zeros(opt_size, dtype=common.type_float)
    t_dir = np.zeros(opt_size, dtype=common.type_float)
    source_up = np.zeros(opt_size, dtype=common.type_float)
    source_dn = np.zeros(opt_size, dtype=common.type_float)
    source_sfc = np.zeros(alb_size, dtype=common.type_float)
    albedo = np.zeros(flx_size, dtype=common.type_float)
    src = np.zeros(flx_size, dtype=common.type_float)
    denom = np.zeros(opt_size, dtype=common.type_float)

    two_stream_args = [ncol, nlay, ngpt, tmin,
            tau, ssa,
            g, mu0,
            r_dif, t_dif,
            r_dir, t_dir,
            t_noscat]

    source_kernel_args = [ncol, nlay, ngpt, top_at_1, r_dir, t_dir, t_noscat,
                          sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir]

    fused_kernel_args = [ncol, nlay, ngpt,
            tau, ssa, g, mu0,
            r_dif, t_dif,
            sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir]

    ref_problem_size = (ncol, ngpt)
    source_kernel_name = f'sw_source_kernel_top_at_1<{common.str_float}, {top_at_1}, {nlay}>'
    source_ref_kernel_name = f'sw_source_kernel<{common.str_float}>'

    two_stream_ref_problem_size = (ncol, nlay, ngpt)
    two_stream_ref_kernel_name = f'sw_2stream_kernel<{common.str_float}>'
    two_stream_kernel_name = f'sw_2stream_kernel<{common.str_float}>'

    fused_kernel_name = f"sw_source_2stream_kernel<{common.str_float}, {top_at_1}>"


    if command_line.tune:
        tune_fused_kernel()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_sw_source_2stream_kernel.json', 'r') as file:
                configurations = json.load(file)
            best_configuration = min(configurations, key=lambda x: x['time'])
            parameters['block_size_x'] = best_configuration['block_size_x']
            parameters['block_size_y'] = best_configuration['block_size_y']
        else:
            parameters['block_size_x'] = command_line.block_size_x
            parameters['block_size_y'] = command_line.block_size_y
        run_and_test(parameters)
