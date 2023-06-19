import logging
import os
import sys

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI
from deephyper.search.hps import MPIDistributedBO

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set the random seed from CL or system clock
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())
FILENAME = f"dtlz_mpi_logs-P/results_seed{SEED}.csv"

# Set default problem parameters
PROB_NUM = "5"
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
from deephyper_benchmark.lib.dtlz import hpo

# Create MPI ranks
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Setup info-level logging
if rank == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - " + \
               "%(message)s",
        force=True,
    )

# define MPI evaluator
evaluator = MPIDistributedBO.bootstrap_evaluator(
    hpo.run,
    evaluator_type="serial",
    storage_type="redis",
    storage_kwargs={
        "host": os.environ.get("DEEPHYPER_DB_HOST", "localhost"),
        "port": 6379,
    },
    comm=comm,
    root=0,
)

# define the search method and scalarization
search = MPIDistributedBO(hpo.problem,
                          evaluator,
                          random_state=SEED,
                          update_prior=True,
                          moo_scalarization_strategy="rPBI",
                          log_dir="dtlz_mpi_logs-P",
                          comm=comm)
# Solve with BB_BUDGET evals
results = search.search(max_evals=BB_BUDGET, timeout=10800)

# Gather performance stats
if rank == 0:
    from deephyper_benchmark.lib.dtlz.metrics import PerformanceEvaluator
    import numpy as np

    # Extract objective values from dataframe
    obj_vals = np.asarray([results["objective_0"].values, results["objective_1"].values, results["objective_2"].values]).T

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
