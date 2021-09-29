import argparse
import json
from collections import OrderedDict
import kernel_tuner as kt
import common


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for add_fluxes_kernel kernel')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=96)
    parser.add_argument('--block_size_y', type=int, default=1)
    parser.add_argument('--block_size_z', type=int, default=1)
    return parser.parse_args()


# Run one instance of the kernel and test output
def run_and_test(params: OrderedDict):
    print(f"Running {kernel_name} [{params['block_size_x']}, {params['block_size_y']}, {params['block_size_z']}]")
    result = kt.run_kernel(kernel_name, kernels_src, problem_size, args, params, compiler_options=common.cp)
    common.compare_fields(flux_up + radn_up, result[6], "flux_up")
    common.compare_fields(flux_dn + radn_dn, result[7], "flux_dn")
    common.compare_fields(flux_up_jac + radn_up_jac, result[8], "flux_up_jac")


# Tuning
def tune():
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(0, 11)]
    tune_params["block_size_y"] = [2**i for i in range(0, 11)]
    tune_params["block_size_z"] = [2**i for i in range(0, 7)]
    restrictions = [f"block_size_x <= {ncol}", f"block_size_y <= {nlev}", f"block_size_z <= {ngpt}"]
    print(f"Tuning {kernel_name}")
    answer = [None for _ in range(0, len(args))]
    answer[6] = flux_up + radn_up
    answer[7] = flux_dn + radn_dn
    answer[8] = flux_up_jac + radn_up_jac
    result, env = kt.tune_kernel(kernel_name, kernels_src, problem_size, args, tune_params, answer=answer,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_add_fluxes_kernel.json", "w") as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    command_line = parse_command_line()

    kernels_src = common.dir_name + "../src_kernels_cuda/rte_solver_kernels.cu"

    # Input
    ncol = common.type_int(512)
    nlay = common.type_int(140)
    nlev = common.type_int(nlay + 1)
    ngpt = common.type_int(224)
    flux_size = ncol * nlev * ngpt
    radn_up = common.random(flux_size, common.type_float)
    radn_dn = common.random(flux_size, common.type_float)
    radn_up_jac = common.random(flux_size, common.type_float)
    # Output
    flux_up = common.random(flux_size, common.type_float)
    flux_dn = common.random(flux_size, common.type_float)
    flux_up_jac = common.random(flux_size, common.type_float)

    kernel_name = f"add_fluxes_kernel<{common.str_float}>"
    problem_size = (ncol, nlev, ngpt)
    args = [ncol, nlev, ngpt, radn_up, radn_dn, radn_up_jac, flux_up, flux_dn, flux_up_jac]

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = OrderedDict()
        if command_line.best_configuration:
            best_configuration = common.best_configuration("timings_add_fluxes_kernel.json")
            parameters["block_size_x"] = best_configuration["block_size_x"]
            parameters["block_size_y"] = best_configuration["block_size_y"]
            parameters["block_size_z"] = best_configuration["block_size_z"]
        else:
            parameters["block_size_x"] = command_line.block_size_x
            parameters["block_size_y"] = command_line.block_size_y
            parameters["block_size_z"] = command_line.block_size_z
        run_and_test(parameters)
