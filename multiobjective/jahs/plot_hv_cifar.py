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
for PNAME in ["AC10", "AC20", "AC40", "random", "C", "L", "P", "Q"]:
    hv_all = []
    bbf_all = []
    # Read results from CSV file
    DNAME = "jahs_mpi_logs-" + PNAME + "-cifar"
    for SEED in range(10):
        FNAME = FILENAME.replace("SEED", str(SEED))
        try:
            results = pandas.read_csv(f"{DNAME}/{FNAME}")
            if PNAME in ["random"]:
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
        plt.plot(np.asarray(bbf_all).mean(axis=0),
                 np.asarray(hv_all).mean(axis=0), "-",
                 label=f"deephyper-{PNAME}")

# And add pymoo to plot
DNAME = "pymoo-cifar"
hv_all = []
bbf_all = []
for SEED in range(10):
    FNAME = FILENAME.replace("SEED", str(SEED))
    hv_vals = []
    bbf_num = []
    obj_vals = []
    try:
        with open(f"{DNAME}/{FNAME}", "r") as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                obj_vals.append([float(fi) for fi in row])
                if i > 0 and (i+1) % 10 == 0:
                    hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
                    bbf_num.append(i+1)
                    assert (len(obj_vals) == bbf_num[-1])
            if len(obj_vals) % 10 != 0:
                hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
                bbf_num.append((len(obj_vals)+9)//10 * 10)
            start = bbf_num[-1]
            for i in range(start, 1001, 10):
                hv_vals.append(hv_vals[-1])
                bbf_num.append(i)
            hv_all.append(hv_vals)
            bbf_all.append(bbf_num)
    except FileNotFoundError:
        print(f"skipping pymoo/NSGA-II seed {SEED}")
if len(bbf_all) > 0:
    # Add to plot
    plt.plot(np.asarray(bbf_all).mean(axis=0),
             np.asarray(hv_all).mean(axis=0), "-",
             label=f"pymoo/NSGA-II")

# And add parmoo + axy to plot
DNAME = "parmoo-axy-cifar"
hv_all = []
bbf_all = []
for SEED in range(10):
    FNAME = FILENAME.replace("SEED", str(SEED))
    hv_vals = []
    bbf_num = []
    obj_vals = []
    try:
        with open(f"{DNAME}/{FNAME}", "r") as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i > 0 and i <= 1000:
                    obj_vals.append([float(fi) for fi in row[-2:]])
                if i > 0 and i <= 1000 and i % 10 == 0:
                    hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
                    bbf_num.append(len(obj_vals))
        while len(hv_vals) < 100:
            hv_vals.append(hv_vals[-1])
        while len(bbf_num) < 100:
            bbf_num.append(bbf_num[-1])
        hv_all.append(hv_vals)
        bbf_all.append(bbf_num)
    except FileNotFoundError:
        print(f"skipping parmoo-axy seed {SEED}")
if len(bbf_all) > 0:
    # Add to plot
    plt.plot(np.asarray(bbf_all).mean(axis=0),
             np.asarray(hv_all).mean(axis=0), "-",
             label=f"parmoo-AXY")

# And add parmoo + rbf to plot
DNAME = "parmoo-rbf-cifar-X"
hv_all = []
bbf_all = []
for SEED in range(10):
    FNAME = FILENAME.replace("SEED", str(SEED))
    hv_vals = []
    bbf_num = []
    obj_vals = []
    try:
        with open(f"{DNAME}/{FNAME}", "r") as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i > 0:
                    obj_vals.append([float(fi) for fi in row[-2:]])
                if i > 10 and (i - 1) % 10 == 0:
                    hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
                    bbf_num.append(len(obj_vals))
        hv_all.append(hv_vals)
        bbf_all.append(bbf_num)
    except FileNotFoundError:
        print(f"skipping parmoo-rbf seed {SEED}")
if len(bbf_all) > 0:
    # Add to plot
    plt.plot(np.asarray(bbf_all).mean(axis=0),
             np.asarray(hv_all).mean(axis=0), "-",
             label=f"parmoo-RBF")

# And add parmoo + tr to plot
DNAME = "parmoo-tr-cifar-X"
hv_all = []
bbf_all = []
for SEED in range(10):
    FNAME = FILENAME.replace("SEED", str(SEED))
    hv_vals = []
    bbf_num = []
    obj_vals = []
    try:
        with open(f"{DNAME}/{FNAME}", "r") as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i > 0:
                    obj_vals.append([float(fi) for fi in row[-2:]])
                if i > 10 and (i - 1) % 10 == 0:
                    hv_vals.append(hypervolume(pareto_front(np.asarray(obj_vals)), nadir))
                    bbf_num.append(len(obj_vals))
            hv_all.append(hv_vals)
            bbf_all.append(bbf_num)
    except FileNotFoundError:
        print(f"skipping parmoo-tr seed {SEED}")
if len(bbf_all) > 0:
    # Add to plot
    plt.plot(np.asarray(bbf_all).mean(axis=0),
             np.asarray(hv_all).mean(axis=0),
             label=f"parmoo-TR")

# Add legends and show
plt.xlabel("Number of blackbox function evaluations")
plt.ylabel("Total hypervolume dominated")
plt.legend(loc="lower right")
plt.tight_layout()
#plt.show()
plt.savefig("cifar_hv.png")

