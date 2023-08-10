from ast import literal_eval
import csv
from deephyper.skopt.moo import pareto_front, hypervolume
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas

FILENAME = "results/jahs-dhquc-40-0/results.csv"
try:
    # Read results from file
    results = pandas.read_csv(FILENAME)
    #valid_loss = np.asarray([100 - np.asarray([literal_eval(fi)]).flatten()[0] for fi in
    #                         results.sort_values("job_id")["objective_0"].values])
    #pred_latency = np.asarray([-np.asarray([literal_eval(fi)]).flatten()[-1] for fi in
    #                           results.sort_values("job_id")["objective_1"].values])
    valid_loss = results.sort_values("job_id")["objective_0"].values
    pred_latency = results.sort_values("job_id")["objective_1"].values
    # Get pareto points
    obj_vals = []
    for i in range(len(valid_loss)):
        obj_vals.append([valid_loss[i], pred_latency[i]])
    pf = pareto_front(np.asarray(obj_vals))
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
plt.show()
#plt.savefig("fig1.svg")
