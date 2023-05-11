
import numpy as np
import pandas as pd
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import EI_RandomConstraint
from parmoo.optimizers import LocalGPS

from parmoo.simulations.dtlz import dtlz2_sim as sim_func # change dtlz1_sim -> dtlz{1,2,3,4,5,6,7}_sim
from parmoo.objectives.obj_lib import single_sim_out

# Turn on logging with timestamps
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Set the random seed from CL or system clock
import sys
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())

### Problem dimensions ###
num_des = 8
num_obj = 3
PROB_NUM = 2

### Budget variables ###
n_search_sz = 10 # 1000 pt initial DOE
n_per_batch = 10  # batch size = 100
iters_limit = 99   # run for 90 iterations


### Start solving problem with RBF surrogate ###

np.random.seed(SEED)

moop_gp = MOOP(LocalGPS)

for i in range(num_des):
    moop_gp.addDesign({'name': f"x{i+1}",
                       'des_type': "continuous",
                       'lb': 0.0, 'ub': 1.0})

moop_gp.addSimulation({'name': "DTLZ_out",
                       'm': num_obj,
                       'sim_func': sim_func(moop_gp.getDesignType(),
                                            num_obj=num_obj, offset=0.5),
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': n_search_sz}})

for i in range(num_obj):
    moop_gp.addObjective({'name': f"f{i+1}",
                          'obj_func': single_sim_out(moop_gp.getDesignType(),
                                                     moop_gp.getSimulationType(),
                                                     ("DTLZ_out", i))})

for i in range(n_per_batch):
   moop_gp.addAcquisition({'acquisition': EI_RandomConstraint,
                           'hyperparams': {}})

# Solve and dump to csv
moop_gp.solve(iters_limit)
results_gp = moop_gp.getObjectiveData(format='pandas')
FILENAME = f"parmoo-gp/results_seed{SEED}.csv"


# Set DTLZ problem environment variables
import os
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = str(num_des)
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = str(num_obj)
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = str(PROB_NUM)
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.5" # [x_o, .., x_d]*=0.5

# Load deephyper performance evaluator
from deephyper_benchmark.lib.DTLZ.metrics import PerformanceEvaluator

# Collect performance stats
obj_vals = []
hv_vals = []
rmse_vals = []
bbf_num = []
perf_eval = PerformanceEvaluator()
for i, row in results_gp.iterrows():
    obj_vals.append([row['f1'], row['f2'], row['f3']])
    if (i+1) > 99 and (i+1) % 100 == 0:
        bbf_num.append(len(obj_vals))
        rmse_vals.append(perf_eval.rmse(np.asarray(obj_vals)))
        hv_vals.append(perf_eval.hypervolume(np.asarray(obj_vals)))

# Dump results to csv file
import csv
with open(FILENAME, "w") as fp:
    writer = csv.writer(fp)
    writer.writerow(bbf_num)
    writer.writerow(hv_vals)
    writer.writerow(rmse_vals)
