#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export NUM_THREADS=1
export OMP_NUM_THREADS=4
export OMP_NESTED=false
export PYTHONPATH=$PYTHONPATH:/home/tchang

#python3 parmoo_solve_axy.py 0
python3 parmoo_solve_rbf.py 0
