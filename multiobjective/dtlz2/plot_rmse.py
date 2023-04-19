from ast import literal_eval
import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas

FILENAME = "results.csv"
PROB_NUM = "2"
BB_BUDGET = 10000 # 10K eval budget
NDIMS = 8 # 8 vars
NOBJS = 3 # 3 objs

# Set DTLZ problem environment variables
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = str(NDIMS)
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = str(NOBJS)
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = PROB_NUM # DTLZ2 problem
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.5" # [x_o, .., x_d]*=0.5

import deephyper_benchmark as dhb
dhb.load("DTLZ")
from deephyper_benchmark.lib.dtlz.metrics import PerformanceEvaluator

# Gather performance stats
for PNAME in ["AC", "C", "L", "P", "Q"]:
    # Read results from CSV file
    DNAME = "dtlz_mpi_logs-" + PNAME
    results = pandas.read_csv(f"{DNAME}/{FILENAME}")
    obj_vals = np.asarray([literal_eval(fi) for fi in
                           results.sort_values("job_id")["objective"].values])
    # Initialize performance arrays
    hv_vals = []
    bbf_num = []
    # Create a performance evaluator for this problem and loop over budgets
    perf_eval = PerformanceEvaluator()
    for i in range(100, BB_BUDGET, 100):
        hv_vals.append(perf_eval.rmse(obj_vals[:i, :]))
        bbf_num.append(i)
    # Don't forget final budget
    hv_vals.append(perf_eval.rmse(obj_vals))
    bbf_num.append(BB_BUDGET)
    # Add to plot
    plt.plot(bbf_num, hv_vals, "-o",
             label=f"deephyper-{PNAME}")

# And add pymoo to plot
DNAME = "pymoo"
hv_vals = []
bbf_num = []
obj_vals = []
with open(f"{DNAME}/{FILENAME}", "r") as fp:
    reader = csv.reader(fp)
    for row in reader:
        for fi in row:
            obj_vals.append([float(fij) for fij in fi.strip()[1:-1].split()])
        hv_vals.append(perf_eval.rmse(np.asarray(obj_vals)))
        bbf_num.append(len(obj_vals))
plt.plot(bbf_num, hv_vals, "-o", label="pymoo/NSGA-II")

# And add parmoo + axy to plot
DNAME = "parmoo-axy"
hv_vals = []
bbf_num = []
obj_vals = []
with open(f"{DNAME}/{FILENAME}", "r") as fp:
    reader = csv.reader(fp)
    for i, row in enumerate(reader):
        if i > 0:
            obj_vals.append([float(fi) for fi in row[-3:]])
        if i > 100 and (i - 1) % 100 == 0:
            hv_vals.append(perf_eval.rmse(np.asarray(obj_vals)))
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
            obj_vals.append([float(fi) for fi in row[-3:]])
        if i > 100 and (i - 1) % 100 == 0:
            hv_vals.append(perf_eval.rmse(np.asarray(obj_vals)))
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
            obj_vals.append([float(fi) for fi in row[-3:]])
        if i > 100 and (i - 1) % 100 == 0:
            hv_vals.append(perf_eval.rmse(np.asarray(obj_vals)))
            bbf_num.append(len(obj_vals))
plt.plot(bbf_num, hv_vals, "-o", label="parmoo-local")

# Add legends and show
plt.xlabel("Number of blackbox function evaluations")
plt.ylabel("RMSE over all residuals")
plt.legend()
plt.tight_layout()
plt.show()

