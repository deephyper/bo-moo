import logging
import os
import sys

# Set the random seed from CL or system clock
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())
FILENAME = f"jahs_mpi_logs-dh/results_seed{SEED}.csv"

# Set default problem parameters
BB_BUDGET = 10000 # 10K eval budget

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

# Setup info-level logging
if rank == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - " + \
               "%(message)s",
        force=True,
    )

from deephyper_benchmark.lib.JAHSBench import hpo

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
                          moo_scalarization_strategy="rChebyshev",
                          log_dir="jahs_mpi_logs-dh",
                          random_state=SEED,
                          comm=comm)

# Solve with BB_BUDGET evals, 3 hr limit
results = search.search(max_evals=BB_BUDGET, timeout=10800)
results.to_csv(f"jahs_mpi_logs-dh/results_seed{SEED}.csv")
