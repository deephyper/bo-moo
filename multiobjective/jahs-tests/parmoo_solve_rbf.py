
import numpy as np
import pandas as pd
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint, FixedWeights
from parmoo.optimizers import LocalGPS

from jahs_bench.api import Benchmark
from parmoo.objectives.obj_lib import single_sim_out

# Turn on logging with timestamps
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Set the random seet from CL or system clock
import sys
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())
FILENAME = f"parmoo-rbf-cifar/results_seed{SEED}.csv"
np.random.seed(SEED)

### Problem dimensions ###
num_des = 8
num_obj = 2

### Budget variables ###
n_search_sz = 200 # 200 pt initial DOE
n_per_batch = 16 # batch size = 16
iters_limit = 50 # run for 50 iterations

### JAHS bench settings ###
DATASET = "cifar10"
MODEL_PATH = "."
NEPOCHS = 200
N_ITERATIONS = 100
# Define the benchmark
benchmark = Benchmark(
        task=DATASET,
        save_dir=MODEL_PATH,
        kind="surrogate",
        download=True
    )


def sim_func(x):
    """ ParMOO compatible simulation function wrapping jahs-bench.

    Args:
        x (numpy structured array): Configuration array with same keys
            as jahs-bench.

    Returns:
        numpy.ndarray: 1D array of size 2 containing the validation loss
        (100 - accuracy) and latency.

    """

    # Default config
    config = {
        'Optimizer': 'SGD',
        'N': 5,
        'W': 16,
        'Resolution': 1.0,
    }
    # Update config using x
    for key in x.dtype.names:
        config[key] = x[key]
    # Special rule for setting "TrivialAugment"
    if x['TrivialAugment'] == "on":
        config['TrivialAugment'] = True
    else:
        config['TrivialAugment'] = False
    # Evaluate and return
    sx = np.zeros(2)
    result = benchmark(config, nepochs=NEPOCHS)
    sx[0] = 100 - result[NEPOCHS]['valid-acc']
    sx[1] = result[NEPOCHS]['latency']
    return sx


### Start solving problem with RBF surrogate ###

np.random.seed(SEED)
moop_rbf = MOOP(LocalGPS)
# 2 continuous variables
moop_rbf.addDesign({'name': "LearningRate",
                    'des_type': "continuous",
                    'lb': 1.0e-3, 'ub': 1.0})
moop_rbf.addDesign({'name': "WeightDecay",
                    'des_type': "continuous",
                    'lb': 1.0e-5, 'ub': 1.0e-3})
# 2 categorical variables
moop_rbf.addDesign({'name': "Activation",
                    'des_type': "categorical",
                    'levels': ["ReLU", "Hardswish", "Mish"]})
moop_rbf.addDesign({'name': "TrivialAugment",
                    'des_type': "categorical",
                    'levels': ["on", "off"]})
# 6 integer variables
for i in range(1, 7):
    moop_rbf.addDesign({'name': f"Op{i}",
                        'des_type': "integer",
                        'lb': 0, 'ub': 4})
# JAHS benchmark
moop_rbf.addSimulation({'name': "jahs",
                        'm': num_obj,
                        'sim_func': sim_func,
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'hyperparams': {'search_budget': n_search_sz}})
# Minimize 2 objectives
moop_rbf.addObjective({'name': "valid-loss",
                       'obj_func': single_sim_out(moop_rbf.getDesignType(),
                                                  moop_rbf.getSimulationType(),
                                                  ("jahs", 0))})
moop_rbf.addObjective({'name': "latency",
                       'obj_func': single_sim_out(moop_rbf.getDesignType(),
                                                  moop_rbf.getSimulationType(),
                                                  ("jahs", 1))})
# q acquisition functions, 1 fixed
weights = np.ones(2)
weights[1] = 1.0e2
moop_rbf.addAcquisition({'acquisition': FixedWeights,
                         'hyperparams': {'weights': weights}})
for i in range(n_per_batch-1):
    moop_rbf.addAcquisition({'acquisition': RandomConstraint,
                             'hyperparams': {}})
# Solve and dump to csv
moop_rbf.solve(iters_limit)
results_rbf = moop_rbf.getObjectiveData(format='pandas')
results_rbf.to_csv(FILENAME)


### Start solving problem with TR iterations ###
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF

FILENAME = f"parmoo-tr-cifar/results_seed{SEED}.csv"
moop_tr = MOOP(TR_LBFGSB)
# 2 continuous variables
moop_tr.addDesign({'name': "LearningRate",
                   'des_type': "continuous",
                   'lb': 1.0e-3, 'ub': 1.0})
moop_tr.addDesign({'name': "WeightDecay",
                   'des_type': "continuous",
                   'lb': 1.0e-5, 'ub': 1.0e-3})
# 2 categorical variables
moop_tr.addDesign({'name': "Activation",
                   'des_type': "categorical",
                   'levels': ["ReLU", "Hardswish", "Mish"]})
moop_tr.addDesign({'name': "TrivialAugment",
                   'des_type': "categorical",
                   'levels': ["on", "off"]})
# 6 integer variables
for i in range(1, 7):
    moop_tr.addDesign({'name': f"Op{i}",
                       'des_type': "integer",
                       'lb': 0, 'ub': 4})
# JAHS benchmark
moop_tr.addSimulation({'name': "jahs",
                       'm': num_obj,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': LocalGaussRBF,
                       'hyperparams': {'search_budget': n_search_sz}})
# Minimize 2 objectives
moop_tr.addObjective({'name': "valid-loss",
                      'obj_func': single_sim_out(moop_rbf.getDesignType(),
                                                 moop_rbf.getSimulationType(),
                                                 ("jahs", 0))})
moop_tr.addObjective({'name': "latency",
                      'obj_func': single_sim_out(moop_rbf.getDesignType(),
                                                 moop_rbf.getSimulationType(),
                                                 ("jahs", 1))})
# 1 acquisition functions, 1 fixed
weights = np.ones(2)
weights[1] = 1.0e3
moop_tr.addAcquisition({'acquisition': FixedWeights,
                        'hyperparams': {'weights': weights}})
for i in range(n_per_batch-1):
    moop_tr.addAcquisition({'acquisition': RandomConstraint,
                            'hyperparams': {}})
# Solve and dump to csv
moop_tr.solve(iters_limit)
results_rbf = moop_tr.getObjectiveData(format='pandas')
results_rbf.to_csv(FILENAME)



## Filter out bad points
#for i, fi in enumerate(results_axy):
#    if any([fi["f1"] > 1, fi["f2"] > 1, fi["f3"] > 1]):
#        results_axy[i]["f1"] = 1.0
#        results_axy[i]["f2"] = 1.0
#        results_axy[i]["f3"] = 1.0
#for i, fi in enumerate(results_rbf):
#    if any([fi["f1"] > 1, fi["f2"] > 1, fi["f3"] > 1]):
#        results_rbf[i]["f1"] = 1.0
#        results_rbf[i]["f2"] = 1.0
#        results_rbf[i]["f3"] = 1.0
#
## We need pymoo to calculate hypervolume indicator of solution set
#from pymoo.indicators.hv import HV
#pts_axy = np.reshape(results_axy["f1"], (results_axy["f1"].size, 1))
#for i in range(1, num_obj):
#    pts_axy = np.concatenate((pts_axy, np.reshape(results_axy[f"f{i+1}"],
#                                                  (results_axy["f1"].size, 1))), axis=1)
#pts_rbf = np.reshape(results_rbf["f1"], (results_rbf["f1"].size, 1))
#for i in range(1, num_obj):
#    pts_rbf = np.concatenate((pts_rbf, np.reshape(results_rbf[f"f{i+1}"],
#                                                  (results_rbf["f1"].size, 1))), axis=1)
## Calculate reference point based on solutions
#rp = np.ones(num_obj) # * np.max(np.concatenate((pts_axy, pts_rbf), axis=0))
#hv = HV(ref_point=rp)
## Initialize plotly trace arrays
#hv_axy = []
#hv_rbf = []
#iter_count = [i for i in range(iters_limit)]
## Now iterate over all iterations
#total_budget = n_search_sz + n_per_batch * iters_limit
#for i in range(n_search_sz, total_budget, n_per_batch):
#    hv_axy.append(hv(pts_axy[:i, :])) # / np.prod(rp))
#    hv_rbf.append(hv(pts_rbf[:i, :])) # / np.prod(rp))
#
#### Generate plotly graphs of results ###
#
#fig = go.Figure()
## Add performance lines
#fig.add_trace(go.Scatter(x=iter_count, y=hv_axy, mode='lines',
#                         name="ParMOO + AXY surrogate",
#                         line=dict(color="blue", width=2),
#                         showlegend=True))
#fig.add_trace(go.Scatter(x=iter_count, y=hv_rbf, mode='lines',
#                         name="ParMOO + RBF surrogate",
#                         line=dict(color="red", width=2),
#                         showlegend=True))
## Set the figure style/layout
#fig.update_layout(
#    xaxis=dict(title="iteration",
#               showline=True,
#               showgrid=True,
#               showticklabels=True,
#               linecolor='rgb(204, 204, 204)',
#               linewidth=2,
#               ticks='outside',
#               tickfont=dict(family='Arial', size=12)),
#    yaxis=dict(title="% total hypervolume",
#               showline=True,
#               showgrid=True,
#               showticklabels=True,
#               linecolor='rgb(204, 204, 204)',
#               linewidth=2,
#               ticks='outside',
#               tickfont=dict(family='Arial', size=12)),
#    plot_bgcolor='white', width=500, height=300,
#    margin=dict(l=80, r=50, t=20, b=20))
#fig.show()
