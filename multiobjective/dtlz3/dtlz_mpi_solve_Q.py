import logging
import os
import sys

# Setup info-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - " + \
           "%(message)s",
    force=True,
)

from mpi4py import MPI
from deephyper.search.hps import MPIDistributedBO

# Set default problem parameters
FILENAME = "results.csv"
PROB_NUM = "3"
BB_BUDGET = 10000 # 10K eval budget
NDIMS = 8 # 8 vars
NOBJS = 3 # 3 objs

# If present, read problem definition from CL
if len(sys.argv) > 1:
    assert sys.argv[1] in [str(i) for i in range(1, 8)]
    PROB_NUM = sys.argv[1]
if len(sys.argv) > 2:
    assert int(sys.argv[2]) > 0
    NDIMS = int(sys.argv[2])
if len(sys.argv) > 3:
    assert int(sys.argv[3]) > 0 and int(sys.argv[3]) < NDIMS
    NOBJS = int(sys.argv[3])
if len(sys.argv) > 4:
    assert int(sys.argv[4]) > 0
    BB_BUDGET = int(sys.argv[4])

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
                          moo_scalarization_strategy="rQuadratic",
                          log_dir="dtlz_mpi_logs-Q",
                          comm=comm)
# Solve with BB_BUDGET evals
results = search.search(max_evals=BB_BUDGET, timeout=10800)

# Gather performance stats
if rank == 0:
    from ast import literal_eval
    from deephyper_benchmark.lib.dtlz.metrics import PerformanceEvaluator
    import numpy as np

    # Extract objective values from dataframe
    obj_vals = np.asarray([literal_eval(fi)
                           for fi in results["objective"].values])

    # Initialize performance arrays
    rmse_vals = []
    npts_vals = []
    hv_vals = []
    bbf_num = []
    # Create a performance evaluator for this problem and loop over budgets
    perf_eval = PerformanceEvaluator()
    for i in range(10, BB_BUDGET, 10):
        hv_vals.append(perf_eval.hypervolume(obj_vals[:i, :]))
        rmse_vals.append(perf_eval.rmse(obj_vals[:i, :]))
        npts_vals.append(perf_eval.numPts(obj_vals[:i, :]))
        bbf_num.append(i)
    # Don't forget final budget
    hv_vals.append(perf_eval.hypervolume(obj_vals))
    rmse_vals.append(perf_eval.rmse(obj_vals))
    npts_vals.append(perf_eval.numPts(obj_vals))
    bbf_num.append(BB_BUDGET)

    # Dump results to csv file
    import csv
    with open(FILENAME, "a") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Problem name", "Method", "metric", "input vars", "objectives"]
                        + bbf_num)
        writer.writerow([f"DLTZ{PROB_NUM}", "DH Quadratic", "hypervol", str(NDIMS), str(NOBJS)]
                         + hv_vals)
        writer.writerow([f"DLTZ{PROB_NUM}", "DH Quadratic", "RMSE", str(NDIMS), str(NOBJS)]
                         + rmse_vals)
        writer.writerow([f"DLTZ{PROB_NUM}", "DH Quadratic", "num pts", str(NDIMS), str(NOBJS)]
                         + npts_vals)
