import csv
from matplotlib import pyplot as plt
import numpy as np

CONF_BOUND = True

# Dirs, names, and colors
dirs = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]
subdirs = ["dtlz_mpi_logs-AC", "dtlz_mpi_logs-C", "dtlz_mpi_logs-L",
           "dtlz_mpi_logs-P", "dtlz_mpi_logs-Q", "pymoo",
           "parmoo-tr"]
labels = ["DeepHyper AugCheb", "DeepHyper Cheb", "DeepHyper Linear",
          "DeepHyper PBI", "DeepHyper Quad", "NSGA-II (pymoo)",
          "ParMOO TR"]
colors = ["g", "r", "b", "c", "m", "y", "violet"]

# Gather performance stats
for di, DNAME in enumerate(subdirs):
    bbf_num = []
    hv_vals = []
    rmse_vals = []
    for iseed in range(10):
        FNAME = f"{DNAME}/results_seed{iseed}.csv"
        for DIR in dirs:
            avg_bbf = []
            avg_rmse = []
            avg_hv = []
            try:
                with open(f"{DIR}/{FNAME}", "r") as fp:
                    csv_reader = csv.reader(fp)
                    avg_bbf.append([float(x) for x in csv_reader.__next__()])
                    avg_hv.append([float(x) for x in csv_reader.__next__()])
                    avg_rmse.append([float(x) for x in csv_reader.__next__()])
                bbf_num.append(np.mean(np.asarray(avg_bbf), axis=0).tolist())
                hv_vals.append(np.mean(np.asarray(avg_hv), axis=0).tolist())
                rmse_vals.append(np.mean(np.asarray(avg_rmse), axis=0).tolist())
            except FileNotFoundError:
                print(f"skipping {DIR}/{FNAME}")
    # Check how many results found
    n = len(bbf_num)
    # Plot mean
    bbf_mean = np.mean(np.array(bbf_num), axis=0)
    hv_mean = np.mean(np.array(hv_vals), axis=0)
    rmse_mean = np.mean(np.array(rmse_vals), axis=0)
    plt.plot(bbf_mean, hv_mean, "-", color=f"{colors[di]}", label=labels[di])
    # If more than 1 result, plot std devs
    if n > 1 and CONF_BOUND:
        hv_std = np.std(np.array(hv_vals), axis=0) / np.sqrt(10)
        rmse_std = np.std(np.array(rmse_vals), axis=0) / np.sqrt(10)
        #plt.fill_between(bbf_mean, hv_mean - 1.96 * hv_std,
        #                 hv_mean + 1.96 * hv_std,
        #                 color=f"{colors[di]}", alpha=0.2)

# Add legends and show
plt.xlabel("Number of blackbox function evaluations")
plt.ylabel("Total hypervolume dominated")
plt.legend(loc="lower right")
plt.tight_layout()
#plt.show()
plt.savefig("dtlz_hv_full.png")
