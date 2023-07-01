#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug 
# #PBF -q prod
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
export NWORKERS=$NDEPTH*$NRANKS_PER_NODE # Number of libE workers

#export log_dir="parmoo-tr"
export PYTHONPATH=$PYTHONPATH:/home/tchang

#mkdir -p $log_dir

# Run the parmoo script
mpiexec -n 3 \
    --envall \
    python lhs_sample.py 0

