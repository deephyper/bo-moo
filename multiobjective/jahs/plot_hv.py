from ast import literal_eval
import csv
from deephyper.skopt.moo import pareto_front, hypervolume
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas

FILENAME = "results.csv"
BB_BUDGET = 100 # 100 eval budget

nadir = np.asarray([100.0, 10.0])

# Gather performance stats
for PNAME in ["AC", "C", "L", "P", "Q"]:
    # Read results from CSV file
    DNAME = "dtlz_mpi_logs-" + PNAME
    try:
        results = pandas.read_csv(f"{DNAME}/{FILENAME}")
        obj_vals = np.asarray([literal_eval(fi) for fi in
                               results.sort_values("job_id")["objective"].values])
        # Initialize performance arrays
        hv_vals = []
        bbf_num = []
        for i in range(10, BB_BUDGET, 10):
            hv_vals.append(hypervolume(pareto_front(obj_vals[:i, :]), nadir))
            bbf_num.append(i)
        # Don't forget final budget
        hv_vals.append(hypervolume(pareto_front(obj_vals), nadir))
        bbf_num.append(BB_BUDGET)
        # Add to plot
        plt.plot(bbf_num, hv_vals, "-o",
                 label=f"deephyper-{PNAME}")
    except FileNotFoundError:
        print(f"skipping deephyper-{PNAME}")

# And add pymoo to plot
DNAME = "pymoo"
hv_vals = []
bbf_num = []
obj_vals = []
try:
    with open(f"{DNAME}/{FILENAME}", "r") as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            for fi in row:
                obj_vals.append([float(fij) for fij in fi.strip()[1:-1].split()])
            hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
            bbf_num.append((i+1)*10)
    plt.plot(bbf_num, hv_vals, "-o", label="pymoo/NSGA-II")
except FileNotFoundError:
    print("skipping pymoo/NSGA-II")

# And add parmoo + axy to plot
DNAME = "parmoo-axy"
hv_vals = []
bbf_num = []
obj_vals = []
with open(f"{DNAME}/{FILENAME}", "r") as fp:
    reader = csv.reader(fp)
    for i, row in enumerate(reader):
        if i > 0:
            obj_vals.append([float(fi) for fi in row[-2:]])
        if i > 10 and (i - 1) % 10 == 0:
            hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
            bbf_num.append(len(obj_vals))
plt.plot(bbf_num, hv_vals, "-o", label="parmoo-AXY")
# And add parmoo + rbf to plot
DNAME = "parmoo-rbf"
hv_vals = []
bbf_num = []
obj_vals = []
with open(f"{DNAME}/{FILENAME}", "r") as fp:
    reader = csv.reader(fp)
    for i, row in enumerate(reader):
        if i > 0:
            obj_vals.append([float(fi) for fi in row[-2:]])
        if i > 10 and (i - 1) % 10 == 0:
            hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
            bbf_num.append(len(obj_vals))
plt.plot(bbf_num, hv_vals, "-o", label="parmoo-RBF")
# And add parmoo + tr to plot
DNAME = "parmoo-tr"
hv_vals = []
bbf_num = []
obj_vals = []
with open(f"{DNAME}/{FILENAME}", "r") as fp:
    reader = csv.reader(fp)
    for i, row in enumerate(reader):
        if i > 0:
            obj_vals.append([float(fi) for fi in row[-2:]])
        if i > 10 and (i - 1) % 10 == 0:
            hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
            bbf_num.append(len(obj_vals))
plt.plot(bbf_num, hv_vals, "-o", label="parmoo-local")

# Add legends and show
plt.xlabel("Number of blackbox function evaluations")
plt.ylabel("Total hypervolume dominated")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

