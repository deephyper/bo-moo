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

from deephyper_benchmark.lib.JAHSBench import hpo

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
    lambda job: hpo.run(job, sleep=False),
    random_state=SEED,
    log_dir=LOG_DIR,
    sampler="NSGAII",
    storage=storage,
    comm=comm,
    n_objectives=3,
)
results = search.search(max_evals=10000, timeout=10000)

MPI.Finalize()
