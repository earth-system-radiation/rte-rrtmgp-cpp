#!/usr/bin/env python
from collections import OrderedDict
import kernel_tuner as kt
import numpy as np
import argparse
import json
import os

from kernel_tuner.integration import store_results
from common import reg_observer

import matplotlib.pyplot as pl
pl.close('all')


# Path to the RCEMIP bins
bin_path = '../rcemip'


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description="Tuning script for Tau_absorption_kernel()")
    parser.add_argument("--tune", default=False, action="store_true")
    parser.add_argument("--run", default=False, action="store_true")
    parser.add_argument("--best_configuration", default=False, action="store_true")
    parser.add_argument("--major_block_size_x", type=int, default=14)
    parser.add_argument("--major_block_size_y", type=int, default=1)
    parser.add_argument("--major_block_size_z", type=int, default=32)

    parser.add_argument("--minor_block_size_x", type=int, default=4)
    parser.add_argument("--minor_block_size_y", type=int, default=4)
    return parser.parse_args()


# Compare results against reference
def compare_fields(arr1, arr2, name):
    okay = np.allclose(arr1, arr2, atol=1e-15)
    max_diff = np.abs(arr1-arr2).max()
    if okay:
        print('results for {}: OKAY!'.format(name))
    else:
        print('results for {}: NOT OKAY, max diff={}'.format(name, max_diff))
        pl.figure()
        pl.plot(arr1-arr2)
        pl.show()


# Run one instance of the kernel and test output
def run_and_test(params: dict):
    # Major
    print("Running {} [block_size_x: {}, block_size_y: {}, block_size_z: {}]".format(
            kernel_name_major,
            params["major_block_size_x"],
            params["major_block_size_y"],
            params["major_block_size_z"]))

    params_major = {'block_size_x': params["major_block_size_x"],
                    'block_size_y': params["major_block_size_y"],
                    'block_size_z': params["major_block_size_z"]}

    result = kt.run_kernel(
        kernel_name_major, kernel_string, problem_size_major,
        args_major, params_major, compiler_options=cp)

    compare_fields(result[-2], tau_after_major, 'major')

    # Minor
    print("Running {} [block_size_x: {}, block_size_y: {}]".format(
            kernel_name_minor, params["minor_block_size_x"], params["minor_block_size_y"]))

    params_minor = {'block_size_x': params["minor_block_size_x"],
                    'block_size_y': params["minor_block_size_y"]}

    # Use output from major as input for minor
    tau[:] = tau_after_major

    result = kt.run_kernel(
        kernel_name_minor, kernel_string, problem_size_minor,
        args_minor_lower, params_minor, compiler_options=cp)

    tau[:] = result[-2]

    result = kt.run_kernel(
        kernel_name_minor, kernel_string, problem_size_minor,
        args_minor_upper, params_minor, compiler_options=cp)

    compare_fields(result[-2], tau_after_minor, 'minor')


# Tuning
def tune():
    params_major = dict()
    params_major["block_size_x"] = list(np.arange(1,4)) #[i for i in range(1, 32 + 1)]
    params_major["block_size_y"] = list(np.arange(1,4)) #[i for i in range(1, 32 + 1)]
    params_major["block_size_z"] = list(np.arange(1,4)) #[i for i in range(1, 32 + 1)]

    params_minor = dict()
    params_minor["block_size_x"] = [i for i in range(1, 32 + 1)]
    params_minor["block_size_y"] = [i for i in range(1, 32 + 1)]
    #params_minor["block_size_x"] = list(np.arange(1,5)) #[i for i in range(1, 32 + 1)]
    #params_minor["block_size_y"] = list(np.arange(1,5)) #[i for i in range(1, 32 + 1)]

    answer_major = len(args_major) * [None]
    answer_major[-2] = tau_after_major

    answer_minor = len(args_minor_lower) * [None]
    answer_minor[-2] = tau_after_minor

    # Reset input tau
    tau[:] = 0.

    #print(f"Tuning {kernel_name_major}")
    #result, env = kt.tune_kernel(
    #    kernel_name_major, kernel_string, problem_size_major,
    #    args_major, params_major, compiler_options=cp,
    #    answer=answer_major, atol=1e-14)

    # This gives an error: `TypeError: Object of type int64 is not JSON serializable`
    #with open("timings_compute_tau_major.json", 'w') as fp:
    #    json.dump(result, fp)

    tau[:] = tau_after_major

    metrics = OrderedDict()
    metrics["registers"] = lambda p: p["num_regs"]

    args = {0: args_minor_upper, 1: args_minor_lower}

    for idx_tropo in [type_int(0), type_int(1)]:

        tau_minor_tropo_one = kt.run_kernel(
            kernel_name_minor, kernel_string, problem_size_minor,
            args[idx_tropo], {"block_size_x": 4, "block_size_y": 4}, compiler_options=cp)
        answer_minor[-2] = tau_minor_tropo_one[-2]

        print(f"Tuning {kernel_name_minor} tropo={idx_tropo}")
        result, env = kt.tune_kernel(
            kernel_name_minor, kernel_string, problem_size_minor,
            args[idx_tropo], params_minor, compiler_options=cp,
            answer=answer_minor, atol=1e-14,
            verbose=True, observers=[reg_observer], metrics=metrics)

        with open(f"timings_compute_tau_minor_{idx_tropo}.json", 'w') as fp:
            json.dump(result, fp)


if __name__ == "__main__":
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
    cp = ['-I{}'.format(include), "-Xptxas=-v"]

    ncol = type_int(512)
    nlay = type_int(140)
    nband = type_int(16)
    ngpt = type_int(256)
    nflav = type_int(10)
    neta = type_int(9)
    npres = type_int(59)
    ntemp = type_int(14)
    ngas = type_int(7)

    nscale_lower = type_int(44)
    nscale_upper = type_int(19)
    nminor_lower = type_int(44)
    nminor_upper = type_int(19)
    nminork_lower = type_int(704)
    nminork_upper = type_int(304)
    idx_h2o = type_int(1)

    # Kernel input
    gpoint_flavor = np.fromfile('{}/gpoint_flavor.bin'.format(bin_path), dtype=type_int)
    band_lims_gpt = np.fromfile('{}/band_lims_gpt.bin'.format(bin_path), dtype=type_int)
    jeta = np.fromfile('{}/jeta.bin'.format(bin_path), dtype=type_int)
    jtemp = np.fromfile('{}/jtemp.bin'.format(bin_path), dtype=type_int)
    jpress = np.fromfile('{}/jpress.bin'.format(bin_path), dtype=type_int)
    minor_limits_gpt_lower = np.fromfile('{}/minor_limits_gpt_lower.bin'.format(bin_path), dtype=type_int)
    minor_limits_gpt_upper = np.fromfile('{}/minor_limits_gpt_upper.bin'.format(bin_path), dtype=type_int)
    idx_minor_lower = np.fromfile('{}/idx_minor_lower.bin'.format(bin_path), dtype=type_int)
    idx_minor_upper = np.fromfile('{}/idx_minor_upper.bin'.format(bin_path), dtype=type_int)
    idx_minor_scaling_lower = np.fromfile('{}/idx_minor_scaling_lower.bin'.format(bin_path), dtype=type_int)
    idx_minor_scaling_upper = np.fromfile('{}/idx_minor_scaling_upper.bin'.format(bin_path), dtype=type_int)
    kminor_start_lower = np.fromfile('{}/kminor_start_lower.bin'.format(bin_path), dtype=type_int)
    kminor_start_upper = np.fromfile('{}/kminor_start_upper.bin'.format(bin_path), dtype=type_int)

    kmajor = np.fromfile('{}/kmajor.bin'.format(bin_path), dtype=type_float)
    col_mix = np.fromfile('{}/col_mix.bin'.format(bin_path), dtype=type_float)
    fmajor = np.fromfile('{}/fmajor.bin'.format(bin_path), dtype=type_float)
    fminor = np.fromfile('{}/fminor.bin'.format(bin_path), dtype=type_float)
    kminor_lower = np.fromfile('{}/kminor_lower.bin'.format(bin_path), dtype=type_float)
    kminor_upper = np.fromfile('{}/kminor_upper.bin'.format(bin_path), dtype=type_float)
    play = np.fromfile('{}/play.bin'.format(bin_path), dtype=type_float)
    tlay = np.fromfile('{}/tlay.bin'.format(bin_path), dtype=type_float)
    col_gas = np.fromfile('{}/col_gas.bin'.format(bin_path), dtype=type_float)

    tropo = np.fromfile('{}/tropo.bin'.format(bin_path), dtype=type_bool)
    minor_scales_with_density_lower = np.fromfile('{}/minor_scales_with_density_lower.bin'.format(bin_path),
                                                  dtype=type_bool)
    minor_scales_with_density_upper = np.fromfile('{}/minor_scales_with_density_upper.bin'.format(bin_path),
                                                  dtype=type_bool)
    scale_by_complement_lower = np.fromfile('{}/scale_by_complement_lower.bin'.format(bin_path), dtype=type_bool)
    scale_by_complement_upper = np.fromfile('{}/scale_by_complement_upper.bin'.format(bin_path), dtype=type_bool)

    # Kernel tuner kernel output
    tau = np.zeros(ngpt * nlay * ncol, dtype=type_float)
    tau_major = np.zeros(ngpt * nlay * ncol, dtype=type_float)
    tau_minor = np.zeros(ngpt * nlay * ncol, dtype=type_float)

    # Reference kernel output
    tau_after_minor = np.fromfile('{}/tau_after_minor.bin'.format(bin_path), dtype=type_float)
    tau_after_major = np.fromfile('{}/tau_after_major.bin'.format(bin_path), dtype=type_float)

    args_major = [
        ncol, nlay, nband, ngpt,
        nflav, neta, npres, ntemp,
        gpoint_flavor,
        band_lims_gpt,
        kmajor,
        col_mix, fmajor,
        jeta, tropo,
        jtemp, jpress,
        tau, tau_major]

    idx_tropo = type_int(1)

    args_minor_lower = [
        ncol, nlay, ngpt,
        ngas, nflav, ntemp, neta,
        nscale_lower,
        nminor_lower,
        nminork_lower,
        idx_h2o, idx_tropo,
        gpoint_flavor,
        kminor_lower,
        minor_limits_gpt_lower,
        minor_scales_with_density_lower,
        scale_by_complement_lower,
        idx_minor_lower,
        idx_minor_scaling_lower,
        kminor_start_lower,
        play,
        tlay,
        col_gas,
        fminor,
        jeta,
        jtemp,
        tropo,
        tau,
        tau_minor]

    idx_tropo = type_int(0)

    args_minor_upper = [
        ncol, nlay, ngpt,
        ngas, nflav, ntemp, neta,
        nscale_upper,
        nminor_upper,
        nminork_upper,
        idx_h2o, type_int(0),
        gpoint_flavor,
        kminor_upper,
        minor_limits_gpt_upper,
        minor_scales_with_density_upper,
        scale_by_complement_upper,
        idx_minor_upper,
        idx_minor_scaling_upper,
        kminor_start_upper,
        play,
        tlay,
        col_gas,
        fminor,
        jeta,
        jtemp,
        tropo,
        tau,
        tau_minor]

    problem_size_major = (ncol, nlay, nband)
    kernel_name_major = 'compute_tau_major_absorption_kernel<{}>'.format(str_float)
    problem_size_minor = (ncol, nlay)
    kernel_name_minor = 'compute_tau_minor_absorption_kernel<{}>'.format(str_float)

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open("timings_compute_tau_major.json", "r") as file:
                configurations_major = json.load(file)
            best_configuration = min(configurations_major, key=lambda x: x["time"])
            parameters['major_block_size_x'] = best_configuration["block_size_x"]
            parameters['major_block_size_y'] = best_configuration["block_size_y"]
            parameters['major_block_size_z'] = best_configuration["block_size_z"]
            with open("timings_compute_tau_minor.json", "r") as file:
                configurations_minor = json.load(file)
            best_configuration = min(configurations_minor, key=lambda x: x["time"])
            parameters['minor_block_size_x'] = best_configuration["block_size_x"]
            parameters['minor_block_size_y'] = best_configuration["block_size_y"]
        else:
            parameters['major_block_size_x'] = command_line.major_block_size_x
            parameters['major_block_size_y'] = command_line.major_block_size_y
            parameters['major_block_size_z'] = command_line.major_block_size_z
            parameters['minor_block_size_x'] = command_line.minor_block_size_x
            parameters['minor_block_size_y'] = command_line.minor_block_size_y
        run_and_test(parameters)
