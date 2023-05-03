#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug 
##PBF -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# source ../../../build/activate-dhenv.sh
source /home/tchang/dh-workspace/scalable-bo/build/activate-dhenv.sh

# Configuration to place 1 worker per GPU
export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH

export log_dir="dtlz_mpi_logs-Q"
export REDIS_CONF="/home/tchang/dh-workspace/scalable-bo/build/redis.conf"

mkdir -p $log_dir

# Setup Redis Database
pushd $log_dir
redis-server $REDIS_CONF &
export DEEPHYPER_DB_HOST=$HOST
popd

sleep 5

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python dtlz_mpi_solve_Q.py 9
