import argparse
import collections
import json
from collections import OrderedDict

import numpy
import numpy as np
import kernel_tuner as kt
import common


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for lw_solver_noscat kernels')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x_step1', type=int, default=32)
    parser.add_argument('--block_size_y_step1', type=int, default=1)
    parser.add_argument('--block_size_z_step1', type=int, default=1)
    parser.add_argument('--block_size_x_step2', type=int, default=32)
    parser.add_argument('--block_size_y_step2', type=int, default=1)
    parser.add_argument('--loop_unroll_step2', type=int, default=0)
    parser.add_argument('--block_size_x_step3', type=int, default=32)
    parser.add_argument('--block_size_y_step3', type=int, default=1)
    parser.add_argument('--block_size_z_step3', type=int, default=1)
    return parser.parse_args()


# Run one instance of the kernel and test output
def run_and_test(params: dict):
    # Step 1
    print('Running {} [block_size_x: {}, block_size_y: {}, block_size_z: {}]'.format(
        kernel_name["step1"],
        params["step1"]['block_size_x'],
        params["step1"]['block_size_y'],
        params["step1"]['block_size_z']))
    ref_result = kt.run_kernel(kernel_name["step1"], ref_kernels_src, problem_size["step1"], ref_args,
                               dict(block_size_x=32, block_size_y=1, block_size_z=1),
                               compiler_options=common.cp)
    for item in range(0, len(ref_args)):
        ref_args[item] = ref_result[item]
    result = kt.run_kernel(kernel_name["step1"], kernels_src, problem_size["step1"], args, params["step1"],
                           compiler_options=common.cp)
    for item in range(0, len(args)):
        args[item] = result[item]
    # Step 2
    print('Running {} [block_size_x: {}, block_size_y: {}, loop_unroll_factor_nlay: {}]'.format(
        kernel_name["step2"],
        params["step2"]['block_size_x'],
        params["step2"]['block_size_y'],
        params["step2"]['loop_unroll_factor_nlay']))
    ref_result = kt.run_kernel(kernel_name["step2"], ref_kernels_src, problem_size["step2"], ref_args,
                               dict(block_size_x=32, block_size_y=1),
                               compiler_options=common.cp)
    for item in range(0, len(ref_args)):
        ref_args[item] = ref_result[item]
    result = kt.run_kernel(kernel_name["step2"], kernels_src, problem_size["step2"], args, params["step2"],
                           compiler_options=common.cp)
    for item in range(0, len(args)):
        args[item] = result[item]
    # Step 3
    print('Running {} [block_size_x: {}, block_size_y: {}, block_size_z: {}]'.format(
        kernel_name["step3"],
        params["step3"]['block_size_x'],
        params["step3"]['block_size_y'],
        params["step3"]['block_size_z']))
    ref_result = kt.run_kernel(kernel_name["step3"], ref_kernels_src, problem_size["step3"], ref_args,
                               dict(block_size_x=32, block_size_y=1, block_size_z=1),
                               compiler_options=common.cp)
    result = kt.run_kernel(kernel_name["step3"], kernels_src, problem_size["step3"], args, params["step3"],
                           compiler_options=common.cp)

    common.compare_fields(ref_result[13], result[13], "radn_up")
    common.compare_fields(ref_result[14], result[14], "radn_dn")
    common.compare_fields(ref_result[16], result[16], "radn_up_jac")


# Tuning
def tuning_verify(reference, tuned_answer, atol=None):
    retval = True
    if atol is None:
        atol = 1e-6
    for index in range(0, len(reference)):
        if reference[index] is None:
            continue
        if index == 14:
            # special case for radn_dn
            for col in range(0, ncol):
                for gpt in range(0, ngpt):
                    if top_at_1:
                        item = col + (gpt * ncol * (nlay + 1))
                    else:
                        item = col + (nlay * ncol) + (gpt * ncol * (nlay + 1))
                    if abs(reference[index][item] - tuned_answer[index][item] > atol):
                        retval = False
                        break
        else:
            retval = numpy.allclose(reference[index], tuned_answer[index], atol=atol)
        if not retval:
            break
    return retval


def tune():
    # Step 1
    print("Tuning {}".format(kernel_name["step1"]))
    ref_result = kt.run_kernel(kernel_name["step1"], ref_kernels_src, problem_size["step1"], ref_args,
                               dict(block_size_x=32, block_size_y=1, block_size_z=1),
                               compiler_options=common.cp)
    answer = [None for _ in range(0, len(args))]
    answer[17] = ref_result[17]
    answer[18] = ref_result[18]
    answer[19] = ref_result[19]
    answer[20] = ref_result[20]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_z"] = [2 ** i for i in range(0, 7)]
    restrictions = [f"block_size_x <= {ncol}", f"block_size_y <= {nlay}", f"block_size_z <= {ngpt}"]
    result, env = kt.tune_kernel(kernel_name["step1"], kernels_src, problem_size["step1"], args, tune_params,
                                 answer=answer, atol=1e-4, verify=tuning_verify,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_lw_solver_noscat_step1.json", "w") as fp:
        json.dump(result, fp)
    for item in range(0, len(ref_args)):
        ref_args[item] = ref_result[item]
    configuration = common.best_configuration("timings_lw_solver_noscat_step1.json")
    result = kt.run_kernel(kernel_name["step1"], kernels_src, problem_size["step1"], args,
                           configuration,
                           compiler_options=common.cp)
    for item in range(0, len(args)):
        args[item] = result[item]
    # Step 2
    print("Tuning {}".format(kernel_name["step2"]))
    ref_result = kt.run_kernel(kernel_name["step2"], ref_kernels_src, problem_size["step2"], ref_args,
                               dict(block_size_x=32, block_size_y=1),
                               compiler_options=common.cp)
    answer = [None for _ in range(0, len(args))]
    answer[13] = ref_result[13]
    answer[14] = ref_result[14]
    answer[16] = ref_result[16]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(0, 11)]
    tune_params["loop_unroll_factor_nlay"] = [0] + [i for i in range(1, nlay + 1) if nlay // i == nlay / i]
    restrictions = [f"block_size_x <= {ncol}", f"block_size_y <= {ngpt}"]
    result, env = kt.tune_kernel(kernel_name["step2"], kernels_src, problem_size["step2"], args, tune_params,
                                 answer=answer, atol=1e-4, verify=tuning_verify,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_lw_solver_noscat_step2.json", "w") as fp:
        json.dump(result, fp)
    for item in range(0, len(ref_args)):
        ref_args[item] = ref_result[item]
    configuration = common.best_configuration("timings_lw_solver_noscat_step2.json")
    result = kt.run_kernel(kernel_name["step2"], kernels_src, problem_size["step2"], args,
                           configuration,
                           compiler_options=common.cp)
    for item in range(0, len(args)):
        args[item] = result[item]
    # Step 3
    print("Tuning {}".format(kernel_name["step3"]))
    ref_result = kt.run_kernel(kernel_name["step3"], ref_kernels_src, problem_size["step3"], ref_args,
                               dict(block_size_x=32, block_size_y=1, block_size_z=1),
                               compiler_options=common.cp)
    answer = [None for _ in range(0, len(args))]
    answer[13] = ref_result[13]
    answer[14] = ref_result[14]
    answer[16] = ref_result[16]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_z"] = [2 ** i for i in range(0, 7)]
    restrictions = [f"block_size_x <= {ncol}", f"block_size_y <= {nlay + 1}", f"block_size_z <= {ngpt}"]
    result, env = kt.tune_kernel(kernel_name["step3"], kernels_src, problem_size["step3"], args, tune_params,
                                 answer=answer, atol=1e-4, verify=tuning_verify,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_lw_solver_noscat_step3.json", "w") as fp:
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
    sfc_alb_dif = np.random.random(1).astype(common.type_float)
    r_dif = np.random.random(1).astype(common.type_float)
    t_dif = np.random.random(1).astype(common.type_float)
    flux_dir = np.random.random(1).astype(common.type_float)
    tau_loc = np.random.random(opt_size).astype(common.type_float)
    trans = np.random.random(opt_size).astype(common.type_float)
    source_dn = np.random.random(opt_size).astype(common.type_float)
    source_up = np.random.random(opt_size).astype(common.type_float)
    sfc_albedo = np.random.random(alb_size).astype(common.type_float)
    source_sfc = np.random.random(alb_size).astype(common.type_float)
    source_sfc_jac = np.random.random(alb_size).astype(common.type_float)
    # Output
    radn_dn = np.random.random(flx_size).astype(common.type_float)
    radn_up = np.random.random(flx_size).astype(common.type_float)
    radn_up_jac = np.random.random(flx_size).astype(common.type_float)

    kernel_name = dict()
    kernel_name["step1"] = "lw_solver_noscat_step1_kernel<{}>".format(common.str_float)
    kernel_name["step2"] = "lw_solver_noscat_step2_kernel<{}>".format(common.str_float)
    kernel_name["step3"] = "lw_solver_noscat_step3_kernel<{}>".format(common.str_float)
    problem_size = dict()
    problem_size["step1"] = (ncol, nlay, ngpt)
    problem_size["step2"] = (ncol, ngpt)
    problem_size["step3"] = (ncol, nlay + 1, ngpt)
    ref_args = [ncol, nlay, ngpt, eps, top_at_1, d, weight, tau, lay_source, lev_source_inc, lev_source_dec,
                sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans,
                source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac]
    args = [ncol, nlay, ngpt, eps, top_at_1, d, weight, tau, lay_source, lev_source_inc, lev_source_dec,
            sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans, source_dn, source_up,
            source_sfc, sfc_albedo, source_sfc_jac]
    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        parameters["step1"] = dict()
        parameters["step2"] = dict()
        parameters["step3"] = dict()
        if command_line.best_configuration:
            best_configuration = common.best_configuration("timings_lw_solver_noscat_step1.json")
            parameters["step1"]['block_size_x'] = best_configuration["block_size_x"]
            parameters["step1"]['block_size_y'] = best_configuration["block_size_y"]
            parameters["step1"]['block_size_z'] = best_configuration["block_size_z"]
            best_configuration = common.best_configuration("timings_lw_solver_noscat_step2.json")
            parameters["step2"]['block_size_x'] = best_configuration["block_size_x"]
            parameters["step2"]['block_size_y'] = best_configuration["block_size_y"]
            parameters["step2"]["loop_unroll_factor_nlay"] = best_configuration["loop_unroll_factor_nlay"]
            best_configuration = common.best_configuration("timings_lw_solver_noscat_step3.json")
            parameters["step3"]['block_size_x'] = best_configuration["block_size_x"]
            parameters["step3"]['block_size_y'] = best_configuration["block_size_y"]
            parameters["step3"]['block_size_z'] = best_configuration["block_size_z"]
        else:
            parameters["step1"]['block_size_x'] = command_line.block_size_x_step1
            parameters["step1"]['block_size_y'] = command_line.block_size_y_step1
            parameters["step1"]['block_size_z'] = command_line.block_size_z_step1
            parameters["step2"]['block_size_x'] = command_line.block_size_x_step2
            parameters["step2"]['block_size_y'] = command_line.block_size_y_step2
            parameters["step2"]['loop_unroll_factor_nlay'] = command_line.loop_unroll_step2
            parameters["step3"]['block_size_x'] = command_line.block_size_x_step3
            parameters["step3"]['block_size_y'] = command_line.block_size_y_step3
            parameters["step3"]['block_size_z'] = command_line.block_size_z_step3
        run_and_test(parameters)
