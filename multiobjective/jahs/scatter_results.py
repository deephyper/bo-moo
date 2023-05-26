from ast import literal_eval
import csv
from deephyper.skopt.moo import pareto_front, hypervolume
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas

FILENAME = "jahs_mpi_logs-AC-cifar/results_seed0.csv"
try:
    # Read results from file
    results = pandas.read_csv(FILENAME)
    valid_loss = np.asarray([100 - np.asarray([literal_eval(fi)]).flatten()[0] for fi in
                             results.sort_values("job_id")["objective_0"].values])
    pred_latency = np.asarray([-np.asarray([literal_eval(fi)]).flatten()[-1] for fi in
                               results.sort_values("job_id")["objective_1"].values])
    # Get pareto points
    obj_vals = np.zeros((1000, 2))
    for i in range(1000):
        obj_vals[i, :] = np.array([valid_loss[i], pred_latency[i]])
    pf = pareto_front(obj_vals[:, :])
    # Add to plot
    plt.scatter(valid_loss,
                pred_latency, color="b",
                label=f"All models")
    plt.scatter(pf[:, 0],
                pf[:, 1], color="r",
                label=f"Pareto optimal models")
except FileNotFoundError:
    print(f"could not open {FILENAME}")

# Add legends and show
plt.xlabel("Validation Loss (%)")
plt.ylabel("Prediction Latency (sec)")
plt.legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.savefig("fig1.svg")
