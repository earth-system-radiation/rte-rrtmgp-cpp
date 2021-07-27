import argparse
import json
from collections import OrderedDict
import numpy as np
import kernel_tuner as kt
import common


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for lw_solver_noscat kernels')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=32)
    parser.add_argument('--block_size_y', type=int, default=1)
    parser.add_argument('--block_size_z', type=int, default=1)
    return parser.parse_args()


# Run one instance of the kernel and test output
def run_and_test_step1(params: dict):
    kernel_name = "lw_solver_noscat_step1_kernel<{}>".format(common.str_float)
    print('Running {} [block_size_x: {}, block_size_y: {}, block_size_z: {}]'.format(
            kernel_name,
            params['block_size_x'],
            params['block_size_y'],
            params['block_size_z']))

    ref_result = kt.run_kernel(kernel_name, ref_kernels_src, problem_size, ref_args, params, compiler_options=common.cp)
    result = kt.run_kernel(kernel_name, kernels_src, problem_size, args, params, compiler_options=common.cp)

    common.compare_fields(ref_result[17], result[17], "tau_loc")
    common.compare_fields(ref_result[18], result[18], "trans")
    common.compare_fields(ref_result[19], result[19], "source_dn")
    common.compare_fields(ref_result[20], result[20], "source_up")


def run_and_test_step2(params: dict):
    kernel_name = "lw_solver_noscat_step2_kernel<{}>".format(common.str_float)
    print('Running {} [block_size_x: {}, block_size_y: {}]'.format(
        kernel_name,
        params['block_size_x'],
        params['block_size_y']))

    ref_result = kt.run_kernel(kernel_name, ref_kernels_src, problem_size, ref_args, params, compiler_options=common.cp)
    result = kt.run_kernel(kernel_name, kernels_src, problem_size, args, params, compiler_options=common.cp)

    common.compare_fields(ref_result[13], result[13], "radn_up")
    common.compare_fields(ref_result[14], result[14], "radn_dn")
    common.compare_fields(ref_result[16], result[16], "radn_up_jac")
    common.compare_fields(ref_result[21], result[21], "source_sfc")
    common.compare_fields(ref_result[22], result[22], "sfc_albedo")
    common.compare_fields(ref_result[23], result[23], "source_sfc_jac")


# Tuning
def tune_step1():
    kernel_name = "lw_solver_noscat_step1_kernel<{}>".format(common.str_float)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(1, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(1, 11)]
    tune_params["block_size_z"] = [2 ** i for i in range(1, 7)]

    params = dict()
    params['block_size_x'] = 32
    params['block_size_y'] = 1
    params['block_size_z'] = 1
    ref_result = kt.run_kernel(kernel_name, ref_kernels_src, problem_size, ref_args, params, compiler_options=common.cp)
    answer = len(args) * [None]
    answer[17] = ref_result[17]
    answer[18] = ref_result[18]
    answer[19] = ref_result[19]
    answer[20] = ref_result[20]

    print("Tuning {}".format(kernel_name))

    result, env = kt.tune_kernel(kernel_name, kernels_src, problem_size, args, tune_params, compiler_options=common.cp,
                                 answer=answer, verbose=True)

    with open("timings_lw_solver_noscat_step1.json", "w") as fp:
        json.dump(result, fp)


def tune_step2():
    kernel_name = "lw_solver_noscat_step2_kernel<{}>".format(common.str_float)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2 ** i for i in range(1, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(1, 11)]

    params = dict()
    params['block_size_x'] = 32
    params['block_size_y'] = 1
    ref_result = kt.run_kernel(kernel_name, ref_kernels_src, problem_size, ref_args, params, compiler_options=common.cp)
    answer = len(args) * [None]
    answer[13] = ref_result[13]
    answer[14] = ref_result[14]
    answer[16] = ref_result[16]
    answer[21] = ref_result[21]
    answer[22] = ref_result[22]
    answer[23] = ref_result[23]

    print("Tuning {}".format(kernel_name))

    result, env = kt.tune_kernel(kernel_name, kernels_src, problem_size, args, tune_params, compiler_options=common.cp,
                                 answer=answer, verbose=True)

    with open("timings_lw_solver_noscat_step2.json", "w") as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    command_line = parse_command_line()

    kernels_src = common.dir_name + '../src_kernels_cuda/rte_solver_kernels.cu'
    ref_kernels_src = common.dir_name + 'reference_kernels/rte_solver_kernels.cu'

    # Input
    ncol = common.type_int(512)
    nlay = common.type_int(140)
    ngpt = common.type_int(224)
    eps = common.type_float(1.0)
    top_at_1 = common.type_bool(1)
    opt_size = ncol * nlay * ngpt
    flx_size = ncol * (nlay + 1) * ngpt
    alb_size = ncol * ngpt
    d = np.random.random(1).astype(common.type_float)
    tau = np.random.random(opt_size).astype(common.type_float)
    lay_source = np.random.random(opt_size).astype(common.type_float)
    lev_source_inc = np.random.random(opt_size).astype(common.type_float)
    lev_source_dec = np.random.random(opt_size).astype(common.type_float)
    sfc_emis = np.random.random(alb_size).astype(common.type_float)
    sfc_src = np.random.random(alb_size).astype(common.type_float)
    sfc_src_jac = np.random.random(alb_size).astype(common.type_float)
    weight = np.random.random(1).astype(common.type_float)
    # Output
    tau_loc = np.zeros(opt_size, dtype=common.type_float)
    tau_loc_ref = np.zeros(opt_size, dtype=common.type_float)
    trans = np.zeros(opt_size, dtype=common.type_float)
    trans_ref = np.zeros(opt_size, dtype=common.type_float)
    source_dn = np.zeros(opt_size, dtype=common.type_float)
    source_dn_ref = np.zeros(opt_size, dtype=common.type_float)
    source_up = np.zeros(opt_size, dtype=common.type_float)
    source_up_ref = np.zeros(opt_size, dtype=common.type_float)
    sfc_albedo = np.zeros(alb_size, dtype=common.type_float)
    sfc_albedo_ref = np.zeros(alb_size, dtype=common.type_float)
    source_sfc = np.zeros(alb_size, dtype=common.type_float)
    source_sfc_ref = np.zeros(alb_size, dtype=common.type_float)
    source_sfc_jac = np.zeros(alb_size, dtype=common.type_float)
    source_sfc_jac_ref = np.zeros(alb_size, dtype=common.type_float)
    radn_dn = np.random.random(flx_size).astype(common.type_float)
    radn_dn_ref = radn_dn
    radn_up = np.zeros(flx_size, dtype=common.type_float)
    radn_up_ref = np.zeros(flx_size, dtype=common.type_float)
    radn_up_jac = np.zeros(flx_size, dtype=common.type_float)
    radn_up_jac_ref = np.zeros(flx_size, dtype=common.type_float)

    # Step 1
    problem_size = (ncol, nlay, ngpt)
    ref_args = [ncol, nlay, ngpt, eps, top_at_1, d, weight, tau, lay_source, lev_source_inc, lev_source_dec,
                sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac, radn_up_jac, tau_loc_ref, trans_ref,
                source_dn_ref, source_up_ref, source_sfc, sfc_albedo, source_sfc_jac]
    args = [ncol, nlay, ngpt, eps, top_at_1, d, weight, tau, lay_source, lev_source_inc, lev_source_dec,
            sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans, source_dn, source_up,
            source_sfc, sfc_albedo, source_sfc_jac]
    if command_line.tune:
        tune_step1()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open("timings_lw_solver_noscat_step1.json", "r") as file:
                configurations = json.load(file)
            best_configuration = min(configurations, key=lambda x: x["time"])
            parameters['block_size_x'] = best_configuration["block_size_x"]
            parameters['block_size_y'] = best_configuration["block_size_y"]
            parameters['block_size_z'] = best_configuration["block_size_z"]
        else:
            parameters['block_size_x'] = command_line.block_size_x
            parameters['block_size_y'] = command_line.block_size_y
            parameters['block_size_z'] = command_line.block_size_z
        run_and_test_step1(parameters)
    # Step 2
    problem_size = (ncol, ngpt)
    ref_args = [ncol, nlay, ngpt, eps, top_at_1, d, weight, tau, lay_source, lev_source_inc, lev_source_dec,
                sfc_emis, sfc_src, radn_up_ref, radn_dn_ref, sfc_src_jac, radn_up_jac_ref, tau_loc, trans,
                source_dn, source_up, source_sfc_ref, sfc_albedo_ref, source_sfc_jac_ref]
    args = [ncol, nlay, ngpt, eps, top_at_1, d, weight, tau, lay_source, lev_source_inc, lev_source_dec,
            sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans, source_dn, source_up,
            source_sfc, sfc_albedo, source_sfc_jac]
    if command_line.tune:
        tune_step2()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open("timings_lw_solver_noscat_step2.json", "r") as file:
                configurations = json.load(file)
            best_configuration = min(configurations, key=lambda x: x["time"])
            parameters['block_size_x'] = best_configuration["block_size_x"]
            parameters['block_size_y'] = best_configuration["block_size_y"]
        else:
            parameters['block_size_x'] = command_line.block_size_x
            parameters['block_size_y'] = command_line.block_size_y
        run_and_test_step2(parameters)
