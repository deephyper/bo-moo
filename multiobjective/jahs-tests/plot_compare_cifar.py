from ast import literal_eval
import csv
from deephyper.skopt.moo import pareto_front, hypervolume
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas

FILENAME = "results_seedSEED.csv"
BB_BUDGET = 1000 # 100 eval budget

nadir = np.asarray([100.0, 10.0])

# Gather performance stats
for DNAME in ["dh-new-cifar", "random-cifar", "dh-old-cifar"]:
    hv_all = []
    bbf_all = []
    # Read results from CSV file
    for SEED in range(10):
        FNAME = FILENAME.replace("SEED", str(SEED))
        try:
            results = pandas.read_csv(f"{DNAME}/{FNAME}")
            if DNAME in ["random-cifar"]:
                obj_vals0 = np.asarray([100 - np.asarray([literal_eval(fi)]).flatten()[0] for fi in
                                        results.sort_values("job_id")["objective_0"].values])
                obj_vals1 = np.asarray([-np.asarray([literal_eval(fi)]).flatten()[-1] for fi in
                                        results.sort_values("job_id")["objective_1"].values])
            else:
                obj_vals0 = np.asarray([100 - results.sort_values("job_id")["objective_0"].values]).flatten()
                obj_vals1 = np.asarray([-results.sort_values("job_id")["objective_1"].values]).flatten()
            obj_vals = np.zeros((1000, 2))
            for i in range(1000):
                obj_vals[i, :] = np.array([obj_vals0[i], obj_vals1[i]])
            # Initialize performance arrays
            hv_vals = []
            bbf_num = []
            for i in range(10, BB_BUDGET+1, 10):
                hv_vals.append(hypervolume(pareto_front(obj_vals[:i, :]), nadir))
                bbf_num.append(i)
            hv_all.append(hv_vals)
            bbf_all.append(bbf_num)
        except FileNotFoundError:
            print(f"skipping {DNAME}/{FNAME}")
    if len(bbf_all) > 0:
        # Add to plot
        bbf_means = np.asarray(bbf_all).mean(axis=0)
        hv_means = np.asarray(hv_all).mean(axis=0)
        hv_se = np.zeros(hv_means.shape)
        for row in hv_all:
            hv_se += (np.asarray(row) - hv_means) ** 2
        hv_se = np.sqrt(hv_se / 9)
        hv_se = hv_se / np.sqrt(10)
        # Add to plot
        if DNAME == "dh-new-cifar":
            plt.plot(bbf_means, hv_means, "-", color="g",
                     label=f"DeepHyper New Cheb")
            plt.fill_between(bbf_means, hv_means - hv_se, hv_means + hv_se,
                             color="g", alpha=0.2)
        elif DNAME == "dh-old-cifar":
            plt.plot(bbf_means, hv_means, "-", color="r",
                     label=f"DeepHyper Old Cheb")
            plt.fill_between(bbf_means, hv_means - hv_se, hv_means + hv_se,
                             color="r", alpha=0.2)
        else:
            plt.plot(bbf_means, hv_means, "-", color="b",
                     label=f"uniform sampling")
            plt.fill_between(bbf_means, hv_means - hv_se, hv_means + hv_se,
                             color="b", alpha=0.2)

# Add legends and show
plt.xlabel("Number of blackbox function evaluations")
plt.ylabel("Total hypervolume dominated")
plt.legend(loc="lower right")
plt.tight_layout()
#plt.show()
plt.savefig("cifar_compare_hv.png")

