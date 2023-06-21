import logging
import os
import sys
import getpass

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI
from deephyper.search.hps._mpi_doptuna import MPIDistributedOptuna

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]

username = getpass.getuser()
host = os.environ["OPTUNA_DB_HOST"]
storage = f"postgresql://{username}@{host}:5432/hpo"
print(storage)

# Set the random seed from CL or system clock
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime

    SEED = int(datetime.now().timestamp())
FILENAME = f"{LOG_DIR}/results_seed{SEED}.csv"

# Set default problem parameters
PROB_NUM = "2"
BB_BUDGET = 100  # 10K eval budget
NDIMS = 8  # 8 vars
NOBJS = 3  # 3 objs

# Set DTLZ problem environment variables
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = str(NDIMS)
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = str(NOBJS)
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = PROB_NUM  # DTLZ2 problem
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.5"  # [x_o, .., x_d]*=0.5

# Load DTLZ benchmark suite
import deephyper_benchmark as dhb

dhb.load("DTLZ")
from deephyper_benchmark.lib.dtlz import hpo

# Create MPI ranks
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Setup info-level logging
if rank == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - "
        + "%(message)s",
        force=True,
    )

# define the optuna search
search = MPIDistributedOptuna(
    hpo.problem,
    hpo.run,
    random_state=SEED,
    log_dir=LOG_DIR,
    sampler="NSGAII",
    storage=storage,
    comm=comm,
    n_objectives=NOBJS,
)
# Solve with BB_BUDGET evals
results = search.search(max_evals=BB_BUDGET)

# Gather performance stats
if rank == 0:
    from deephyper_benchmark.lib.dtlz.metrics import PerformanceEvaluator
    import numpy as np

    # Extract objective values from dataframe
    obj_vals = np.asarray(
        [
            results["objective_0"].values,
            results["objective_1"].values,
            results["objective_2"].values,
        ]
    ).T

    # Initialize performance arrays
    rmse_vals = []
    npts_vals = []
    hv_vals = []
    bbf_num = []
    # Create a performance evaluator for this problem and loop over budgets
    perf_eval = PerformanceEvaluator()
    for i in range(100, BB_BUDGET, 100):
        hv_vals.append(perf_eval.hypervolume(obj_vals[:i, :]))
        rmse_vals.append(perf_eval.rmse(obj_vals[:i, :]))
        npts_vals.append(perf_eval.numPts(obj_vals[:i, :]))
        bbf_num.append(i)
    # Don't forget final budget
    hv_vals.append(perf_eval.hypervolume(obj_vals))
    rmse_vals.append(perf_eval.rmse(obj_vals))
    bbf_num.append(BB_BUDGET)

    # Dump results to csv file
    import csv

    with open(FILENAME, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(bbf_num)
        writer.writerow(hv_vals)
        writer.writerow(rmse_vals)

MPI.Finalize()
