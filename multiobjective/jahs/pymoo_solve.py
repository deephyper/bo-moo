import csv
import numpy as np
import os
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
import sys

from jahs_bench.api import Benchmark

# Set the random seed from CL or system clock
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())
FILENAME = f"pymoo/results_seed{SEED}.csv"

# Set default problem parameters
PROB_NUM = "2"
BB_BUDGET = 10000 # 10K eval budget
NDIMS = 8 # 8 vars
NOBJS = 3 # 3 objs

# Set DTLZ problem environment variables
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = str(NDIMS)
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = str(NOBJS)
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = PROB_NUM # DTLZ2 problem
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.5" # [x_o, .., x_d]*=0.5

# Load DTLZ benchmark suite
import deephyper_benchmark as dhb
dhb.load("DTLZ")
from deephyper_benchmark.lib.dtlz.metrics import PerformanceEvaluator

### JAHS bench settings ###
DATASET = "cifar10"
MODEL_PATH = "."
NEPOCHS = 200
N_ITERATIONS = 100
# Define the benchmark
benchmark = Benchmark(
        task=DATASET,
        save_dir=MODEL_PATH,
        kind="surrogate",
        download=True
    )


def sim_func(x):
    """ ParMOO compatible simulation function wrapping jahs-bench.

    Args:
        x (numpy structured array): Configuration array with same keys
            as jahs-bench.

    Returns:
        numpy.ndarray: 1D array of size 2 containing the validation loss
        (100 - accuracy) and latency.

    """

    sx = np.zeros((x.size[0], 2))
    for i, xi in enumerate(x):
        # Default config
        config = {
            'Optimizer': 'SGD',
            'N': 5,
            'W': 16,
            'Resolution': 1.0,
        }
        # Update config using x
        for key in x.dtype.names:
            config[key] = x[key]
        # Special rule for setting "TrivialAugment"
        if x['TrivialAugment'] == "on":
            config['TrivialAugment'] = True
        else:
            config['TrivialAugment'] = False
        # Evaluate and return
        result = benchmark(config, nepochs=NEPOCHS)
        sx[i, 0] = 100 - result[NEPOCHS]['valid-acc']
        sx[i, 1] = result[NEPOCHS]['latency']
    return sx

# Solve DTLZ2 problem w/ NSGA-II (pop size 100) in pymoo
problem = sim_func
algorithm = NSGA2(pop_size=50)
res = minimize(problem,
               algorithm,
               ("n_gen", 20),
               save_history=True,
               seed=SEED,
               verbose=False)

# Extract all objective values
obj_vals = []
for row in res.history:
    for fi in row.result().F:
        obj_vals.append(fi)
obj_vals = np.asarray(obj_vals)

# Dump results to csv file
with open(FILENAME, "w") as fp:
    writer = csv.writer(fp)
    for row in obj_vals:
        writer.writerow(row)
