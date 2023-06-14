#!/bin/bash
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
# #PBS -q debug 
#PBF -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# source ../../../build/activate-dhenv.sh
source /home/tchang/dh-workspace/scalable-bo/build/activate-dhenv.sh

# Configuration to place 1 worker per GPU
export NDEPTH=16 # this * NRANKS_PER_NODE (below) = 64
export NRANKS_PER_NODE=2 # Should be a small number, number of workers per node
export NNODES=`wc -l < $PBS_NODEFILE` # Get number of nodes checked out
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE )) # 25 n * 4 w/n = 100 w
export OMP_NUM_THREADS=$NDEPTH

export log_dir="jahs_mpi_logs-AC"
export REDIS_CONF="/home/tchang/dh-workspace/scalable-bo/build/redis.conf"
export PYTHONPATH=$PYTHONPATH:/home/tchang

mkdir -p $log_dir

# Setup Redis Database
pushd $log_dir
redis-server $REDIS_CONF &
export DEEPHYPER_DB_HOST=$HOST
popd

sleep 10

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 0

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 1

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 2

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 3

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 4

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 5

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 6

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 7

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 8

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_AC.py 9
