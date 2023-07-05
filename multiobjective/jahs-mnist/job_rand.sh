#!/bin/bash
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
##PBS -q debug 
#PBF -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# source ../../../build/activate-dhenv.sh
source /home/tchang/dh-workspace/scalable-bo/build/activate-dhenv.sh

# Configuration to place 1 worker per GPU
export NDEPTH=16 # this * NRANKS_PER_NODE (below) = 64
export NRANKS_PER_NODE=4 # Should be a small number, number of workers per node
export NNODES=`wc -l < $PBS_NODEFILE` # Get number of nodes checked out
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE )) # 25 n * 4 w/n = 100 w
export OMP_NUM_THREADS=$NDEPTH

export log_dir="jahs_mpi_logs-random"
export REDIS_CONF="/home/tchang/dh-workspace/scalable-bo/build/redis.conf"
export PYTHONPATH=$PYTHONPATH:/home/tchang

mkdir -p $log_dir

# Setup Redis Database
pushd $log_dir
redis-server $REDIS_CONF &
export DEEPHYPER_DB_HOST=$HOST
popd

sleep 5

# Set random seed
export SEED=0

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_random.py $SEED
