#!/bin/bash
#PBS -l select=30:system=polaris
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# Activate parmoo+libE venv
module load conda
conda activate base
source /grand/datascience/tchang/polaris/libe-env/bin/activate

#!!! CONFIGURATION - START
# ~~~ EDIT: used to create the name of the experiment folder
# ~~~ you can use the following variables and pass them to your python script
export problem="jahs"
export search="parmoo"
export SEED=1
#!!! CONFIGURATION - END
#

# Configuration to place 1 worker per GPU
export NRANKS_PER_NODE=4 # Should be a small number, number of workers per node
export NNODES=`wc -l < $PBS_NODEFILE` # Get number of nodes checked out
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export NWORKERS=$(( $NTOTRANKS + 1 )) # Number of libE workers

# Set path and log dirs
export PYTHONPATH=$PYTHONPATH:/home/tchang
export log_dir="results/$problem-$search-$NNODES-$SEED"
mkdir -p $log_dir

sleep 50

# Run the parmoo script from inside the log_dir
cd $log_dir
mpiexec -n ${NWORKERS} \
    --envall \
    python ../../polaris_parmoo_test.py $SEED

