#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export NUM_THREADS=1
export OMP_NUM_THREADS=4
export OMP_NESTED=false
export PYTHONPATH=$PYTHONPATH:/home/tchang

python3 parmoo_solve_rbf.py 0
python3 parmoo_solve_rbf.py 1
python3 parmoo_solve_rbf.py 2
python3 parmoo_solve_rbf.py 3
python3 parmoo_solve_rbf.py 4
python3 parmoo_solve_rbf.py 5
python3 parmoo_solve_rbf.py 6
python3 parmoo_solve_rbf.py 7
python3 parmoo_solve_rbf.py 8
python3 parmoo_solve_rbf.py 9
