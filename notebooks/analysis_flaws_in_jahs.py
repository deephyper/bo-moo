"""
Analyzing JAHS Bench 201 datasets...
"""

"""
Import and combine all of our datasets to get our best approximation
to the surrogate problem's PF.
"""

import pandas as pd
import numpy as np
from deephyper.skopt.moo import pareto_front

methods = [
    "random",
    "nsgaii",
    "tpe",
    "dhqu",
    "dhmml",
    "dhquc",
    "dhqu-cheb",
    "dhquc-cheb"
]
sizes = [
    "10",
    "20",
    "30",
    "40"
]
seeds = [f"{i}" for i in range(10)] # Generate 10 random seeds
# Templates for generating searching
results_dir = "../multiobjective/jahs-mnist/results" # Path to the results
file_name = "jahs-METHOD-SIZE-SEED/results.csv"
soln_pts = []
for mi in methods:
    for ni in sizes:
        for si in seeds:
            print(f"trying {results_dir}/{file_name}..."
                  .replace("METHOD", mi)
                  .replace("SIZE", ni)
                  .replace("SEED", si)
            )
            try:
                dfi = pd.read_csv(f"{results_dir}/{file_name}"
                          .replace("METHOD", mi)
                          .replace("SIZE", ni)
                          .replace("SEED", si)
                )
                print("\tfound! Adding pts to list...")
                if "pareto_efficient" in dfi.columns: # Some runs were done before this column was added to deephyper
                    dfi = dfi.loc[dfi["pareto_efficient"]] # Get Pareto points
                for idx, row in dfi.iterrows():
                    if row['objective_0'] >= 88:
                        soln_pts.append([
                            100-row['objective_0'],
                            -row['objective_1'],
                            -row['objective_2']
                        ])
                print("\tdone!")
            except FileNotFoundError:
                print(f"\t{results_dir}/{file_name} does not exist, skipping..."
                      .replace("METHOD", mi)
                      .replace("SIZE", ni)
                      .replace("SEED", si)
                )
print("\tfiltering Pareto front...")
soln_pts = pareto_front(np.asarray(soln_pts))
print(soln_pts.shape)

"""
Show the computed Pareto front for just the first 2 objectives since these
are the ones recommended by JAHS Bench 201 paper -- the third objective (model
size in MB) was only added by us.

There may be some tradeoff, but only at a very small scale compared to the full
range of values.
"""

from matplotlib import pyplot as plt

soln_pts_2d = pareto_front(soln_pts[:, :2])
plt.scatter(soln_pts_2d[:,0], soln_pts_2d[:,1])
plt.xlabel("Error rate (100 - Accuracy)")
plt.ylabel("Latency (secs)")
plt.show()

"""
Show pairwise scatter plots for all objectives.
"""

fig, axs = plt.subplots(2,2)
axs[0,0].scatter(soln_pts[:,0], soln_pts[:,1])
axs[0,0].set_xlabel("Error rate (100 - Accuracy)")
axs[0,0].set_ylabel("Latency (secs)")
axs[1,1].scatter(soln_pts[:,1], soln_pts[:,2])
axs[1,1].set_xlabel("Latency (secs)")
axs[1,1].set_ylabel("Model size (MB)")
axs[1,0].scatter(soln_pts[:,0], soln_pts[:,2])
axs[1,0].set_xlabel("Error rate (100 - Accuracy)")
axs[1,0].set_ylabel("Model size (MB)")
plt.tight_layout()
plt.show()

""" 
The above plot is too difficult to see the tradeoffs.
Zoom in on just the regions of interest for each pair.
"""

fig, axs = plt.subplots(2,2)
axs[0,0].scatter(soln_pts[:,0], soln_pts[:,1])
axs[0,0].set_xbound((4, 10))
axs[0,0].set_ybound((0, 2))
axs[0,0].set_xlabel("Error rate (100 - Accuracy)")
axs[0,0].set_ylabel("Latency (secs)")
axs[1,1].scatter(soln_pts[:,1], soln_pts[:,2])
axs[1,1].set_xbound((0, 2))
axs[1,1].set_ybound((0, 1))
axs[1,1].set_xlabel("Latency (secs)")
axs[1,1].set_ylabel("Model size (MB)")
axs[1,0].scatter(soln_pts[:,0], soln_pts[:,2])
axs[1,0].set_xbound((4, 10))
axs[1,0].set_ybound((0, 1))
axs[1,0].set_xlabel("Error rate (100 - Accuracy)")
axs[1,0].set_ylabel("Model size (MB)")
plt.tight_layout()
plt.show()

"""
Define the Hausdorff metric for calculating the distance between two
discrete PF approximations.

The Hausdorff distance is a common quantification for set differences
in topology. For discrete point-sets, it reduces to the max of
generational distance and inverse generational distance.

"""

def _gd(soln_pts, true_pts):
    gd_sum = 0.0
    for si in soln_pts:
        gd_sum += np.min(np.linalg.norm(true_pts - si, axis=1))
    return gd_sum / soln_pts.shape[0]

def _igd(soln_pts, true_pts):
    igd_sum = 0.0
    for ti in true_pts:
        igd_sum += np.min(np.linalg.norm(soln_pts - ti, axis=1))
    return igd_sum / true_pts.shape[0]

def hausdorff(soln_pts, true_pts):
    """ Compute the Hausdorff distance between two Pareto fronts """
    return max(_gd(soln_pts, true_pts), _igd(soln_pts, true_pts))

## Test the above functions, uncomment to run
#df = pd.read_csv(f"{results_dir}/{file_name}"
#                 .replace("METHOD", "nsgaii")
#                 .replace("SIZE", "40")
#                 .replace("SEED", "0")
#)
#if "pareto_efficient" in df.columns:
#    df = df.loc[dfi["pareto_efficient"]] # Get Pareto points
#soln2 = []
#for idx, row in df.iterrows():
#    if row['objective_0'] >= 88:
#        soln2.append([
#            100-row['objective_0'],
#            -row['objective_1'],
#            -row['objective_2']
#        ])
#soln2 = np.asarray(soln2)
#print(hausdorff(soln2, soln_pts)) # Should be nonzero
#print(hausdorff(soln_pts, soln_pts)) # Should be 0

"""
Import the original JAHS Bench datasets for comparison and extract the
true Pareto front
"""

true_pf = np.random.sample((100, 3)) * 10 # TODO when download finishes
true_pf = pareto_front(true_pf)

"""
Scatter the true PF against the surrogate PF and visually observe the
difference.
"""

fig, axs = plt.subplots(2,2)
axs[0,0].scatter(soln_pts[:,0], soln_pts[:,1])
axs[0,0].scatter(true_pf[:,0], true_pf[:,1], color="r")
axs[0,0].set_xbound((4, 10))
axs[0,0].set_ybound((0, 2))
axs[0,0].set_xlabel("Error rate (100 - Accuracy)")
axs[0,0].set_ylabel("Latency (secs)")
axs[1,1].scatter(soln_pts[:,1], soln_pts[:,2])
axs[1,1].scatter(true_pf[:,1], true_pf[:,2], color="r")
axs[1,1].set_xbound((0, 2))
axs[1,1].set_ybound((0, 1))
axs[1,1].set_xlabel("Latency (secs)")
axs[1,1].set_ylabel("Model size (MB)")
axs[1,0].scatter(soln_pts[:,0], soln_pts[:,2])
axs[1,0].scatter(true_pf[:,0], true_pf[:,2], color="r")
axs[1,0].set_xbound((4, 10))
axs[1,0].set_ybound((0, 1))
axs[1,0].set_xlabel("Error rate (100 - Accuracy)")
axs[1,0].set_ylabel("Model size (MB)")
plt.tight_layout()
plt.show()

"""
Calculate the Hausdorff distance between the two solution sets to
see how close they are.
"""

print(hausdorff(soln_pts, true_pf))
