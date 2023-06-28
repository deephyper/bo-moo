import csv
from matplotlib import pyplot as plt
import numpy as np

CONF_BOUND = True

# Dirs, names, and colors
dirs = ["dtlz_mpi_logs-AC", "dtlz_mpi_logs-C", "dtlz_mpi_logs-L",
        "dtlz_mpi_logs-P", "dtlz_mpi_logs-Q", "pymoo", "parmoo-tr"]
labels = ["DeepHyper AugCheb", "DeepHyper Cheb", "DeepHyper Linear",
          "DeepHyper PBI", "DeepHyper Quad", "NSGA-II (pymoo)",
          "ParMOO TR"]
colors = ["g", "r", "b", "c", "m", "y", "violet"]

# Gather performance stats
for di, DNAME in enumerate(dirs):
    bbf_num = []
    hv_vals = []
    rmse_vals = []
    for iseed in range(10):
        FNAME = f"{DNAME}/results_seed{iseed}.csv"
        try:
            with open(FNAME, "r") as fp:
                csv_reader = csv.reader(fp)
                bbf_num.append([float(x) for x in csv_reader.__next__()])
                hv_vals.append([float(x) for x in csv_reader.__next__()])
                rmse_vals.append([float(x) for x in csv_reader.__next__()])
        except FileNotFoundError:
            print(f"skipping {DNAME}")
    # Check how many results found
    n = len(bbf_num)
    if n > 0:
        # Plot mean
        bbf_mean = np.mean(np.array(bbf_num), axis=0)
        hv_mean = np.mean(np.array(hv_vals), axis=0)
        rmse_mean = np.mean(np.array(rmse_vals), axis=0)
        plt.plot(bbf_mean, rmse_mean, "-", color=f"{colors[di]}", label=labels[di])
        # If more than 1 result, plot std errors
        if n > 1 and CONF_BOUND:
            hv_std = np.std(np.array(hv_vals), axis=0) / np.sqrt(10)
            rmse_std = np.std(np.array(rmse_vals), axis=0) / np.sqrt(10)
            plt.fill_between(bbf_mean, rmse_mean - 1.96 * rmse_std,
                             rmse_mean + 1.96 * rmse_std,
                             color=f"{colors[di]}", alpha=0.2)

# Add legends and show
plt.xlabel("Number of blackbox function evaluations")
plt.ylabel("RMSE of Pareto optimal points")
plt.legend(loc="upper right")
plt.tight_layout()
#plt.show()
plt.savefig("dtlz2_rmse.png")
