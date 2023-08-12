#!/bin/bash
#PBS -l select=40:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
##PBS -q debug 
#PBF -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home
##PBS -J 0-4
##PBS -r y

set -xe

cd ${PBS_O_WORKDIR}

# TODO: Adapt Environment
source /lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/activate-dhenv.sh # Env/Romain
# source /home/tchang/dh-workspace/scalable-bo/build/activate-dhenv.sh # Env/Tyler

#!!! CONFIGURATION - START
# ~~~ EDIT: used to create the name of the experiment folder
# ~~~ you can use the following variables and pass them to your python script
export problem="jahs"
export search="dhquc"
export SEED=${PBS_ARRAY_INDEX}
#!!! CONFIGURATION - END
#

# Configuration to place 1 worker per GPU
export NDEPTH=16 # this * NRANKS_PER_NODE (below) = 64
export NRANKS_PER_NODE=4 # Should be a small number, number of workers per node
export NNODES=`wc -l < $PBS_NODEFILE` # Get number of nodes checked out
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE )) # 25 n * 4 w/n = 100 w
export OMP_NUM_THREADS=$NDEPTH

# Mkdirs / activation files
# TODO: Adapt Redis Conf
# export REDIS_CONF="/home/tchang/dh-workspace/scalable-bo/build/redis.conf" # TODO
export DEEPHYPER_LOG_DIR="results/$problem-$search-cheb-$NNODES-$SEED"
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
    python jahs_mpi_solve_dhquc_cheb.py $SEED
