#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q debug 
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# source ../../../build/activate-dhenv.sh
source ../../../scalable-bo/build/activate-dhenv.sh

# Configuration to place 1 worker per GPU
export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH

export log_dir="mpi-distributed-log"
mkdir -p $log_dir

# Setup Redis Database
pushd $log_dir
redis-server $REDIS_CONF --requirepass "helloworld" &
export DEEPHYPER_DB_HOST=$HOST
popd

sleep 5

# Run the DeepHyper script
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python mpi_dbo_with_redis.py
