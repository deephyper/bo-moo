import logging
import os
import sys

# Set the random seed from CL or system clock
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())

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

LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]

# Setup info-level logging
if rank == 0:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - " + \
               "%(message)s",
        force=True,
    )

# Set the problem name
os.environ["DEEPHYPER_BENCHMARK_JAHS_PROB"] = "fashion_mnist"

import deephyper_benchmark as dhb
dhb.load("JAHSBench")
from deephyper_benchmark.lib.jahsbench import hpo

# define MPI evaluator
evaluator = MPIDistributedBO.bootstrap_evaluator(
    lambda job: hpo.run(job, sleep=True),
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
                          moo_scalarization_strategy="Chebyshev",
                          moo_scalarization_weight="random",
                          # update_prior=True,
                          # update_prior_quantile=0.25,
                          # moo_lower_bounds=[0.9, None, None],
                          objective_scaler="quantile-uniform",
                          #objective_scaler="minmaxlog",
                          verbose=1,
                          log_dir=LOG_DIR,
                          random_state=SEED,
                          comm=comm)

# Solve with 10000 evals, 10000 sec limit
results = search.search(max_evals=10000, timeout=10000)
