#!/bin/bash
#PBS -l select=30:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
##PBS -q debug 
#PBF -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# source ../../../build/activate-dhenv.sh
source /lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/activate-dhenv.sh

#!!! CONFIGURATION - START
# ~~~ EDIT: used to create the name of the experiment folder
# ~~~ you can use the following variables and pass them to your python script
export problem="jahs"
export search="dh"
export SEED=9
#!!! CONFIGURATION - END
#

# Configuration to place 1 worker per GPU
export NDEPTH=16 # this * NRANKS_PER_NODE (below) = 64
export NRANKS_PER_NODE=4 # Should be a small number, number of workers per node
export NNODES=`wc -l < $PBS_NODEFILE` # Get number of nodes checked out
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE )) # 25 n * 4 w/n = 100 w
export OMP_NUM_THREADS=$NDEPTH

# Mkdirs / activation files
export REDIS_CONF="/lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/redis.conf"
export PYTHONPATH=$PYTHONPATH:/home/tchang
export DEEPHYPER_LOG_DIR="results/$problem-$search-$NNODES-$SEED"
mkdir -p $DEEPHYPER_LOG_DIR

# Setup Redis Database
pushd $DEEPHYPER_LOG_DIR
redis-server $REDIS_CONF &
export DEEPHYPER_DB_HOST=$HOST
popd

sleep 5

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    python jahs_mpi_solve_dh.py $SEED
