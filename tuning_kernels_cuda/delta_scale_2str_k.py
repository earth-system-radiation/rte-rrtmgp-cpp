import argparse
import json
from collections import OrderedDict
import kernel_tuner as kt
import numpy as np

import common


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description="Tuning script for delta_scale_2str_k_kernel")
    parser.add_argument("--tune", default=False, action="store_true")
    parser.add_argument("--run", default=False, action="store_true")
    parser.add_argument("--best_configuration", default=False, action="store_true")
    parser.add_argument("--ncol", type=int, default=512)
    parser.add_argument("--nlay", type=int, default=140)
    parser.add_argument("--ngpt", type=int, default=224)
    parser.add_argument("--block_size_x", type=int, default=32)
    parser.add_argument("--block_size_y", type=int, default=16)
    parser.add_argument("--block_size_z", type=int, default=1)
    return parser.parse_args()


# Run one instance of the kernel and test output
def run_and_test(params: OrderedDict):
    print(
        f"Running {kernel_name} [{params['block_size_x']}, {params['block_size_y']}, {params['block_size_z']}]")
    ref_result = kt.run_kernel(kernel_name, kernels_src_ref, problem_size, args_ref,
                               OrderedDict(block_size_x=32, block_size_y=16, block_size_z=1),
                               compiler_options=common.cp)
    result = kt.run_kernel(kernel_name, kernels_src, problem_size, args, params["kernel"],
                           compiler_options=common.cp)
    common.compare_fields(ref_result[4], result[4], "tau")
    common.compare_fields(ref_result[5], result[5], "ssa")
    common.compare_fields(ref_result[6], result[6], "g")


# Tuning
def tune():
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(0, 11)]
    tune_params["block_size_z"] = [2 ** i for i in range(0, 7)]
    restrictions = [f"block_size_x <= {ncol}", f"block_size_y <= {nlay}", f"block_size_z <= {ngpt}"]
    print(f"Tuning {kernel_name}")
    ref_result = kt.run_kernel(kernel_name, kernels_src_ref, problem_size, args_ref,
                               OrderedDict(block_size_x=32, block_size_y=16, block_size_z=1),
                               compiler_options=common.cp)
    answer = [None for _ in range(0, len(args))]
    answer[4] = ref_result[4]
    answer[5] = ref_result[5]
    answer[6] = ref_result[6]
    result, env = kt.tune_kernel(kernel_name, kernels_src, problem_size, args, tune_params, answer=answer,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_delta_scale_2str_k_kernel.json", "w") as fp:
        json.dump(result, fp)


if __name__ == "__main__":
    command_line = parse_command_line()

    kernels_src = common.dir_name + "../src_kernels_cuda/optical_props_kernels.cu"
    kernels_src_ref = common.dir_name + "/reference_kernels/optical_props_kernels.cu"

    # Data
    ncol = common.type_int(command_line.ncol)
    nlay = common.type_int(command_line.nlay)
    ngpt = common.type_int(command_line.ngpt)
    size = ncol * nlay * ngpt
    eps = common.type_float(1.0)
    tau = common.random(size, common.type_float)
    tau_ref = tau
    ssa = common.random(size, common.type_float)
    ssa_ref = ssa
    g = common.random(size, common.type_float)
    g_ref = g

    kernel_name = f"delta_scale_2str_k_kernel<{common.str_float}>"
    problem_size = (ncol, nlay, ngpt)
    args = [ncol, nlay, ngpt, eps, tau, ssa, g]
    args_ref = [ncol, nlay, ngpt, eps, tau_ref, ssa_ref, g_ref]

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = OrderedDict()
        if command_line.best_configuration:
            best_configuration = common.best_configuration("timings_delta_scale_2str_k_kernel.json")
            parameters["block_size_x"] = best_configuration["block_size_x"]
            parameters["block_size_y"] = best_configuration["block_size_y"]
            parameters["block_size_z"] = best_configuration["block_size_z"]
        else:
            parameters["block_size_x"] = command_line.block_size_x
            parameters["block_size_y"] = command_line.block_size_y
            parameters["block_size_z"] = command_line.block_size_z
        run_and_test(parameters)
