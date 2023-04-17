import numpy as np

from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

# Get DTLZ2 problem
problem = get_problem("dtlz2", n_var=8, n_obj=3)

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               save_history=True,
               #seed=1,
               verbose=False)

import csv

with open("pymoo/results.csv", "w") as fp:
    writer = csv.writer(fp)
    for row in res.history:
        writer.writerow(row.result().F)
