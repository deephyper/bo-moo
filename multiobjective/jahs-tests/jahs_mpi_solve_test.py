import logging
import os
import sys

from mpi4py import MPI
from deephyper.search.hps import MPIDistributedBO

# Set the random seed from CL or system clock
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())
FILENAME = f"jahs_mpi_logs-test/results_seed{SEED}.csv"

# Set default problem parameters
BB_BUDGET = 100 # 100 eval budget

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
                          acq_func="UCBd",
                          acq_optimizer="mixedga",
                          acq_optimizer_freq=1,
                          moo_scalarization_strategy="rAugChebyshev",
                          log_dir="jahs_mpi_logs-test",
                          random_state=SEED,
                          comm=comm)

# Solve with BB_BUDGET evals
results = search.search(max_evals=BB_BUDGET, timeout=10800)
results.to_csv(f"jahs_mpi_logs-test/results_seed{SEED}.csv")
