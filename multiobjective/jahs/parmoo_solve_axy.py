
import numpy as np
import pandas as pd
from parmoo import MOOP
from parmoo.searches import LatinHypercube
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
FILENAME = f"parmoo-axy/results_seed{SEED}.csv"
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


### Define the AXY surrogate ###

from tlux.approximate.axy import AXY
from parmoo.structs import SurrogateFunction

class AxySurrogate(SurrogateFunction):
   def __init__(self, m, lb, ub, hyperparams):
      self.model = AXY()
      self.settings = dict(
         m=m,
         lb=lb,
         ub=ub,
         eps=hyperparams.get("des_tol", np.ones(len(lb)) * 0.0001)
      )
      self.x = []
      self.f = []

   # Fit the AXY model.
   def fit(self, x, f):
      self.x.append(x)
      self.f.append(f)
      self.model = AXY(mds=128, mns=4)
      self.model.fit(
         x=np.concatenate(self.x, axis=0),
         y=np.concatenate(self.f, axis=0),
         steps=3000,
         min_update_ratio=1.0,
      )

   # Do a new "fit".
   def update(self, *args, **kwargs):
      return self.fit(*args, **kwargs)

   # Evaluate the model.
   def evaluate(self, x):
      return self.model.predict(x.reshape((1,-1))).flatten()

   # Finite difference approximation to the gradient.
   def gradient(self, x):
      # At X
      points = [x]
      # Near X
      for i in range(len(x)):
         points.append(
            x.copy()
         )
         points[-1][i] += self.settings['eps'][i]
      # Get value
      values = self.model.predict(
         x=np.concatenate(points, axis=0)
      )
      # Linear fit.
      control_points = np.concatenate(points, axis=0)
      # Process and store local information
      ones_column = np.ones( (control_points.shape[0],1) )
      coef_matrix = np.concatenate((control_points, ones_column), axis=1)
      # Returns (model, residuals, rank, singular values), we want model
      weights, residuals = np.linalg.lstsq(coef_matrix, values, rcond=None)[:2]
      print("x.shape: ", x.shape)
      print("self.settings: ", self.settings)
      print("weights.shape: ", weights.shape)
      # Return the gradient.
      return weights[:-1]

   # Return a random point (because we don't know how to improve the given one).
   def improve(self, x, global_improv):
      if global_improv:
         return [np.random.random(self.settings["lb"].shape) * (self.settings["ub"] - self.settings["lb"]) + self.settings["lb"]]
      else:
         return [x + (np.random.random(size=x.shape) * 2 - 1) * np.sqrt(self.settings["eps"])]

   # Determine radius of trust region (default to the entire design space).
   def setCenter(self, x):
      return np.max(self.settings["ub"] - self.settings["lb"])

   # Save model
   def save(self, filename):
      return self.model.save(filename)

   # Load model
   def load(self, filename):
      return self.model.load(filename)

### Start solving problem with AXY surrogate ###

moop_axy = MOOP(LocalGPS)
# 2 continuous variables
moop_axy.addDesign({'name': "LearningRate",
                    'des_type': "continuous",
                    'lb': 1.0e-3, 'ub': 1.0})
moop_axy.addDesign({'name': "WeightDecay",
                    'des_type': "continuous",
                    'lb': 1.0e-5, 'ub': 1.0e-3})
# 2 categorical variables
moop_axy.addDesign({'name': "Activation",
                    'des_type': "categorical",
                    'levels': ["ReLU", "Hardswish", "Mish"]})
moop_axy.addDesign({'name': "TrivialAugment",
                    'des_type': "categorical",
                    'levels': ["on", "off"]})
# 6 integer variables
for i in range(1, 7):
    moop_axy.addDesign({'name': f"Op{i}",
                        'des_type': "integer",
                        'lb': 0, 'ub': 4})
# JAHS benchmark
moop_axy.addSimulation({'name': "jahs",
                        'm': num_obj,
                        'sim_func': sim_func,
                        'search': LatinHypercube,
                        'surrogate': AxySurrogate,
                        'hyperparams': {'search_budget': n_search_sz}})
# Minimize 2 objectives
moop_axy.addObjective({'name': "valid-loss",
                       'obj_func': single_sim_out(moop_axy.getDesignType(),
                                                  moop_axy.getSimulationType(),
                                                  ("jahs", 0))})
moop_axy.addObjective({'name': "latency",
                       'obj_func': single_sim_out(moop_axy.getDesignType(),
                                                  moop_axy.getSimulationType(),
                                                  ("jahs", 1))})
# q acquisition functions, 1 fixed
weights = np.ones(2)
weights[1] = 1.0e2
moop_axy.addAcquisition({'acquisition': FixedWeights,
                         'hyperparams': {'weights': weights}})
for i in range(n_per_batch - 1):
    moop_axy.addAcquisition({'acquisition': RandomConstraint,
                             'hyperparams': {}})
# Solve and dump to csv
moop_axy.solve(iters_limit)
results_axy = moop_axy.getObjectiveData(format='pandas')
results_axy.to_csv(FILENAME)


