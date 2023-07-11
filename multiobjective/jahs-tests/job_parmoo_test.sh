#!/bin/bash
##PBS -l select=30:system=polaris
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

#!!! CONFIGURATION - START
# ~~~ EDIT: used to create the name of the experiment folder
# ~~~ you can use the following variables and pass them to your python script
export problem="jahs"
export search="parmoo"
export SEED=0
#!!! CONFIGURATION - END
#

# Configuration to place 1 worker per GPU
export NDEPTH=16 # this * NRANKS_PER_NODE (below) = 64
export NRANKS_PER_NODE=4 # Should be a small number, number of workers per node
export NNODES=`wc -l < $PBS_NODEFILE` # Get number of nodes checked out
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH
export NWORKERS=$(( $NTOTRANKS + 1 )) # Number of libE workers

# Set path and log dirs
export PYTHONPATH=$PYTHONPATH:/home/tchang
export log_dir="results/$problem-$search-$NNODES-$SEED"
mkdir -p $log_dir

#sleep 50

# Run the parmoo script from inside the log_dir
cd $log_dir
mpiexec -n ${NWORKERS} \
    --envall \
    python ../../polaris_parmoo_test.py $SEED

