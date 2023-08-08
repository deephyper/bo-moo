#!/bin/bash
for SEED in 0 1 2 3 4 5 6 7 8 9
do
	python3 dtlz_mpi_solve_optuna.py $SEED
done
