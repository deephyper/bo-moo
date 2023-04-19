
import numpy as np
import pandas as pd
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.optimizers import LocalGPS
#import plotly.graph_objects as go

from parmoo.simulations.dtlz import dtlz2_sim as sim_func # change dtlz1_sim -> dtlz{1,2,3,4,5,6,7}_sim
from parmoo.objectives.obj_lib import single_sim_out

# Turn on logging with timestamps
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

### Problem dimensions ###
num_des = 8
num_obj = 3

### Budget variables ###
n_search_sz = 1000 # 1000 pt initial DOE
n_per_batch = 100  # batch size = 100
iters_limit = 90   # run for 90 iterations

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
      self.model = AXY(mds=32, mns=4)
      self.model.fit(
         x=np.concatenate(self.x, axis=0),
         y=np.concatenate(self.f, axis=0),
         steps=1000,
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

for i in range(num_des):
    moop_axy.addDesign({'name': f"x{i+1}",
                        'des_type': "continuous",
                        'lb': 0.0, 'ub': 1.0})

moop_axy.addSimulation({'name': "DTLZ_out",
                        'm': num_obj,
                        'sim_func': sim_func(moop_axy.getDesignType(),
                                             num_obj=num_obj, offset=0.5),
                        'search': LatinHypercube,
                        'surrogate': AxySurrogate,
                        'hyperparams': {'search_budget': n_search_sz}})

for i in range(num_obj):
    moop_axy.addObjective({'name': f"f{i+1}",
                           'obj_func': single_sim_out(moop_axy.getDesignType(),
                                                      moop_axy.getSimulationType(),
                                                      ("DTLZ_out", i))})

for i in range(n_per_batch):
   moop_axy.addAcquisition({'acquisition': RandomConstraint,
                            'hyperparams': {}})

# Solve with 10 iteration * 10 candidates per iteration + 100 pt search = 200 sim budget
moop_axy.solve(iters_limit)
results_axy = moop_axy.getObjectiveData(format='pandas')
results_axy.to_csv("parmoo-axy/results.csv")

### Uncomment ParMOO's outputs below ###

## Display solution
#print(results_axy, "\n dtype=" + str(results_axy.dtype))
#
## Plot results -- must have extra viz dependencies installed
#from parmoo.viz import scatter
## The optional arg `output` exports directly to jpg instead of interactive mode
#scatter(moop_axy, output="jpeg")


### Start solving problem with RBF surrogate ###

moop_rbf = MOOP(LocalGPS)

for i in range(num_des):
    moop_rbf.addDesign({'name': f"x{i+1}",
                        'des_type': "continuous",
                        'lb': 0.0, 'ub': 1.0})

moop_rbf.addSimulation({'name': "DTLZ_out",
                        'm': num_obj,
                        'sim_func': sim_func(moop_rbf.getDesignType(),
                                             num_obj=num_obj, offset=0.5),
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'hyperparams': {'search_budget': n_search_sz}})

for i in range(num_obj):
    moop_rbf.addObjective({'name': f"f{i+1}",
                           'obj_func': single_sim_out(moop_rbf.getDesignType(),
                                                      moop_rbf.getSimulationType(),
                                                      ("DTLZ_out", i))})

for i in range(n_per_batch):
   moop_rbf.addAcquisition({'acquisition': RandomConstraint,
                            'hyperparams': {}})

# Solve with 10 iteration * 10 candidates per iteration + 100 pt search = 200 sim budget
moop_rbf.solve(iters_limit)
results_rbf = moop_rbf.getObjectiveData(format='pandas')
results_rbf.to_csv("parmoo-rbf/results.csv")


### Start solving problem with TR iterations ###
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF

moop_tr = MOOP(TR_LBFGSB)

for i in range(num_des):
    moop_tr.addDesign({'name': f"x{i+1}",
                       'des_type': "continuous",
                       'lb': 0.0, 'ub': 1.0})

moop_tr.addSimulation({'name': "DTLZ_out",
                       'm': num_obj,
                       'sim_func': sim_func(moop_tr.getDesignType(),
                                            num_obj=num_obj, offset=0.5),
                       'search': LatinHypercube,
                       'surrogate': LocalGaussRBF,
                       'hyperparams': {'search_budget': n_search_sz}})

for i in range(num_obj):
    moop_tr.addObjective({'name': f"f{i+1}",
                          'obj_func': single_sim_out(moop_tr.getDesignType(),
                                                     moop_tr.getSimulationType(),
                                                     ("DTLZ_out", i))})

for i in range(n_per_batch):
   moop_tr.addAcquisition({'acquisition': RandomConstraint,
                           'hyperparams': {}})

# Solve with 10 iteration * 10 candidates per iteration + 100 pt search = 200 sim budget
moop_tr.solve(iters_limit)
results_rbf = moop_tr.getObjectiveData(format='pandas')
results_rbf.to_csv("parmoo-tr/results.csv")



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
