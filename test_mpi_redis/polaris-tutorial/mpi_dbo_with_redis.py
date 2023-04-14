import os
import logging

logging.basicConfig(
    filename="deephyper.log", # optional if we want to store the logs to disk
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

from mpi4py import MPI

from deephyper.search.hps import MPIDistributedBO

from ackley import hp_problem, run

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each rank creates a RedisStorage client and connects to the storage server
# indicated by host:port. Then, the storage is passed to the evaluator.
evaluator = MPIDistributedBO.bootstrap_evaluator(
   run,
   evaluator_type="serial", # one worker to evaluate the run-function per rank
   storage_type="redis",
   storage_kwargs={
      "host": os.environ.get("DEEPHYPER_DB_HOST", "localhost"),
      "port": 6379,
   },
   comm=comm,
   root=0,
)

# A new search was created by the bootstrap_evaluator function.
if rank == 0:
   logging.info(f"Search Id: {evaluator._search_id}")

# Define the Periodic Exponential Decay scheduler to avoid over-exploration
# When increasing the number of parallel workers. This mechanism will enforce
# All agent to periodically converge to the exploitation regime of Bayesian Optimization.
scheduler = {
         "type": "periodic-exp-decay",
         "periode": 50,
         "rate": 0.1,
}

# The Distributed Bayesian Optimization search instance is created
# With the corresponding evaluator and communicator.
search = MPIDistributedBO(
   hp_problem, 
   evaluator, 
   log_dir="mpi-distributed-log", 
   comm=comm, 
   scheduler=scheduler
)

# The search is started with a timeout of 2 minutes.
results = search.search(timeout=120)
