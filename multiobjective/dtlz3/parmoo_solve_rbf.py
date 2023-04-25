
import numpy as np
import pandas as pd
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.optimizers import LocalGPS

from parmoo.simulations.dtlz import dtlz3_sim as sim_func # change dtlz1_sim -> dtlz{1,2,3,4,5,6,7}_sim
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
n_search_sz = 2000 # 2000 pt initial DOE
n_per_batch = 100  # batch size = 100
iters_limit = 40   # run for 40 iterations


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

# Solve and dump to csv
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

# Solve and dump to csv
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
