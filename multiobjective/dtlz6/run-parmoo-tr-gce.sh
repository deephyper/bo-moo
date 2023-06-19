#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/tchang

#!/bin/bash
for SEED in 0 1 2 3 4 5 6 7 8 9
do
	python3 parmoo_solve_tr.py $SEED
done
