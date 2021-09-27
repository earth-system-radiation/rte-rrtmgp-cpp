from kernel_tuner.observers import BenchmarkObserver
from collections import OrderedDict
import numpy as np
import os
import json

# Settings
type_int = np.int32
type_float = np.float64
type_bool = np.int32  # = default without `RTE_RRTMGP_USE_CBOOL`
str_float = 'float' if type_float is np.float32 else 'double'
np.set_printoptions(edgeitems=50)
# CUDA source code
dir_name = os.path.dirname(os.path.realpath(__file__)) + '/'
# CUDA compiler parameters
include = dir_name + '../include'
cp = [f"-I{include}", "-Wno-deprecated-gpu-targets"]


# get number of registers
class RegisterObserver(BenchmarkObserver):
    def get_results(self):
        return {"num_regs": self.dev.func.num_regs}


# Compare results against reference
def compare_fields(arr1, arr2, name):
    okay = np.allclose(arr1, arr2, atol=1e-15)
    max_diff = np.abs(arr1-arr2).max()

    if okay:
        print('results for {}: OKAY!'.format(name))
    else:
        print('results for {}: NOT OKAY, max diff={}'.format(name, max_diff))


# Return the configuration with minimum execution time
def best_configuration(json_file: str):
    with open(json_file, "r") as file:
        configurations = json.load(file)
    return min(configurations, key=lambda x: x["time"])


# return zero filled array
def zero(size, datatype):
    return np.zeros(size).astype(datatype)


reg_observer = RegisterObserver()
metrics = OrderedDict()
metrics["registers"] = lambda p: p["num_regs"]
