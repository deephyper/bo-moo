#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug 
##PBF -q preemptable
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
export REDIS_CONF="/home/tchang/dh-workspace/scalable-bo/build/redis.conf"

# Set the seed
export seed=9

## Run DeepHyper AC
## Setup Redis Database
#export log_dir="dtlz_mpi_logs-AC"
#mkdir -p $log_dir
#pushd $log_dir
#redis-server $REDIS_CONF &
#export DEEPHYPER_DB_HOST=$HOST
#popd
#sleep 5
## Run the DeepHyper script
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
#    --depth=${NDEPTH} \
#    --cpu-bind depth \
#    --envall \
#    python dtlz_mpi_solve_AC.py $seed
## Stop the redis server
#redis-cli shutdown
#
## Run DeepHyper C
## Setup Redis Database
#export log_dir="dtlz_mpi_logs-C"
#mkdir -p $log_dir
#pushd $log_dir
#redis-server $REDIS_CONF &
#export DEEPHYPER_DB_HOST=$HOST
#popd
#sleep 5
## Run the DeepHyper script
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
#    --depth=${NDEPTH} \
#    --cpu-bind depth \
#    --envall \
#    python dtlz_mpi_solve_C.py $seed
## Stop the redis server
#redis-cli shutdown

# Run DeepHyper L
# Setup Redis Database
export log_dir="dtlz_mpi_logs-L"
mkdir -p $log_dir
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
    python dtlz_mpi_solve_L.py $seed
# Stop the redis server
redis-cli shutdown

# Run DeepHyper P
# Setup Redis Database
export log_dir="dtlz_mpi_logs-P"
mkdir -p $log_dir
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
    python dtlz_mpi_solve_P.py $seed
# Stop the redis server
redis-cli shutdown

# Run DeepHyper Q
# Setup Redis Database
export log_dir="dtlz_mpi_logs-Q"
mkdir -p $log_dir
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
    python dtlz_mpi_solve_Q.py $seed
# Stop the redis server
redis-cli shutdown
