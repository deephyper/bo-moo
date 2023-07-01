#!/bin/bash
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
# #PBS -q debug 
#PBF -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# source ../../../build/activate-dhenv.sh
source /home/tchang/dh-workspace/scalable-bo/build/activate-dhenv.sh

# Random seed
export SEED=0

# Configuration to place 1 worker per GPU
export NDEPTH=16 # this * NRANKS_PER_NODE (below) = 64
export NRANKS_PER_NODE=4 # Should be a small number, number of workers per node
export NNODES=`wc -l < $PBS_NODEFILE` # Get number of nodes checked out
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH
export NWORKERS=$(( $NTOTRANKS + 1 )) # Number of libE workers

export log_dir="parmoo-tr"
export PYTHONPATH=$PYTHONPATH:/home/tchang

mkdir -p $log_dir

# Run the parmoo script
mpiexec -n ${NWORKERS} \
    --envall \
    python parmoo_solve_tr_libe.py $SEED
# Save libE stats for runtime analysis
mv libE_stats.txt $log_dir/stats_seed0.csv

