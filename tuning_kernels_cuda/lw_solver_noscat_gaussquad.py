import argparse
import json
from collections import OrderedDict
import kernel_tuner as kt
import common


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='Tuning script for lw_solver_noscat_gaussquad kernel')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=32)
    parser.add_argument('--block_size_y', type=int, default=1)
    return parser.parse_args()


def run_and_test(params: OrderedDict):
    print(f"Running {kernel_name} [block_size_x: {params['block_size_x']}, block_size_y: {params['block_size_y']}]")
    ref_result = kt.run_kernel(kernel_name, ref_kernels_src, problem_size, ref_args,
                               OrderedDict(block_size_x=32, block_size_y=1), compiler_options=common.cp)
    result = kt.run_kernel(kernel_name, kernels_src, problem_size, args, params, compiler_options=common.cp)
    common.compare_fields(ref_result[25], result[25], "flux_up")
    common.compare_fields(ref_result[26], result[26], "flux_dn")
    common.compare_fields(ref_result[27], result[27], "flux_up_jac")


# Tuning
def tune():
    print(f"Tuning {kernel_name}")
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(0, 11)]
    tune_params["block_size_y"] = [2 ** i for i in range(0, 11)]
    restrictions = [f"block_size_x <= {ncol}", f"block_size_y <= {ngpt}"]
    ref_result = kt.run_kernel(kernel_name, ref_kernels_src, problem_size, ref_args,
                               OrderedDict(block_size_x=32, block_size_y=1), compiler_options=common.cp)
    result, env = kt.tune_kernel(kernel_name, kernels_src, problem_size, args, tune_params, answer=ref_result,
                                 compiler_options=common.cp, verbose=True, restrictions=restrictions)
    with open("timings_lw_solver_noscat_gaussquad.json", "w") as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    command_line = parse_command_line()

    kernels_src = common.dir_name + '../src_kernels_cuda/rte_solver_kernels.cu'
    ref_kernels_src = common.dir_name + 'reference_kernels/rte_solver_kernels.cu'

    # Input
    ncol = common.type_int(512)
    nlay = common.type_int(140)
    ngpt = common.type_int(224)
    opt_size = ncol * nlay * ngpt
    flx_size = ncol * (nlay + 1) * ngpt
    alb_size = ncol * ngpt
    eps = common.type_float(1.0)
    top_at_1 = common.type_bool(1)
    nmus = common.type_int(1)
    ds = common.random(nmus, common.type_float)
    weights = common.random(nmus, common.type_float)
    tau = common.random(opt_size, common.type_float)
    lay_source = common.random(opt_size, common.type_float)
    lev_source_inc = common.random(opt_size, common.type_float)
    lev_source_dec = common.random(opt_size, common.type_float)
    sfc_emis = common.random(alb_size, common.type_float)
    sfc_src = common.random(alb_size, common.type_float)
    radn_up = common.random(flx_size, common.type_float)
    radn_dn = common.random(flx_size, common.type_float)
    sfc_src_jac = common.random(alb_size, common.type_float)
    radn_up_jac = common.random(flx_size, common.type_float)
    tau_loc = common.random(opt_size, common.type_float)
    trans = common.random(opt_size, common.type_float)
    source_dn = common.random(opt_size, common.type_float)
    source_up = common.random(opt_size, common.type_float)
    source_sfc = common.random(alb_size, common.type_float)
    sfc_albedo = common.random(alb_size, common.type_float)
    source_sfc_jac = common.random(alb_size, common.type_float)
    flux_up = common.random(flx_size, common.type_float)
    flux_up_ref = flux_up
    flux_dn = common.random(flx_size, common.type_float)
    flux_dn_ref = flux_dn
    flux_up_jac = common.random(flx_size, common.type_float)
    flux_up_jac_ref = flux_up_jac

    kernel_name = f"lw_solver_noscat_gaussquad_kernel<{common.str_float}>"
    problem_size = (ncol, ngpt)
    ref_args = [ncol, nlay, ngpt, eps, top_at_1, nmus, ds, weights, tau, lay_source, lev_source_inc, lev_source_dec,
                sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans, source_dn, source_up,
                source_sfc, sfc_albedo, source_sfc_jac, flux_up_ref, flux_dn_ref, flux_up_jac_ref]
    args = [ncol, nlay, ngpt, eps, top_at_1, nmus, ds, weights, tau, lay_source, lev_source_inc, lev_source_dec,
            sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans, source_dn, source_up,
            source_sfc, sfc_albedo, source_sfc_jac, flux_up, flux_dn, flux_up_jac]
    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = OrderedDict()
        if command_line.best_configuration:
            best_configuration = common.best_configuration("timings_lw_solver_noscat_gaussquad.json")
            parameters["block_size_x"] = best_configuration["block_size_x"]
            parameters["block_size_y"] = best_configuration["block_size_y"]
        else:
            parameters["block_size_x"] = command_line.block_size_x
            parameters["block_size_y"] = command_line.block_size_y
        run_and_test(parameters)
