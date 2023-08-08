import csv
import matplotlib as mpl


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    From: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


width, height = set_size(252, fraction=1.0)
widht = 2 * width

fontsize = 9

mpl.rcParams.update(
    {
        "font.size": fontsize,
        "figure.figsize": (width, height),
        "figure.facecolor": "white",
        "savefig.dpi": 360,
        "figure.subplot.bottom": 0.125,
        "figure.edgecolor": "white",
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
)

from matplotlib import pyplot as plt
import numpy as np

CONF_BOUND = True

# Dirs, names, and colors
dirs = [
    "dtlz1",
    "dtlz2",
    "dtlz3",
    "dtlz4",
    "dtlz5",
    "dtlz6",
    "dtlz7",
]
subdirs = [
    "dtlz_mpi_logs-AC_qu",
    "dtlz_mpi_logs-C_qu",
    "dtlz_mpi_logs-L_qu",
    "dtlz_mpi_logs-P_qu",
    "dtlz_mpi_logs-Q_qu",
    "pymoo",
    "dtlz_mpi_logs-AC_mml",
    "dtlz_mpi_logs-C_mml",
    "dtlz_mpi_logs-L_mml",
    "dtlz_mpi_logs-P_mml",
    "dtlz_mpi_logs-Q_mml",
    "parmoo-tr",
]
labels = [
    "AC-QU",
    "C-QU",
    "L-QU",
    "PBI-QU",
    "Q-QU",
    "NSGAII",
    "AC-MML",
    "C-MML",
    "L-MML",
    "PBI-MML",
    "Q-MML",
    "TR",
]
colors = [
    "maroon",
    "orange",
    "seagreen",
    "royalblue",
    "purple",
    "red",
    "maroon",
    "orange",
    "seagreen",
    "royalblue",
    "purple",
    "pink",
]
linestyle = [
    "--",
    "--",
    "--",
    "--",
    "--",
    "-",
    ":",
    ":",
    ":",
    ":",
    ":",
    "-",
]

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
    plt.plot(
        bbf_mean,
        rmse_mean,
        linestyle=linestyle[di],
        color=f"{colors[di]}",
        label=labels[di],
    )
    # If more than 1 result, plot std devs
    if n > 1 and CONF_BOUND:
        hv_std = np.std(np.array(hv_vals), axis=0) / np.sqrt(n)
        rmse_std = np.std(np.array(rmse_vals), axis=0) / np.sqrt(n)
        plt.fill_between(
            bbf_mean,
            rmse_mean - 1.96 * rmse_std,
            rmse_mean + 1.96 * rmse_std,
            color=f"{colors[di]}",
            alpha=0.2,
        )

# Add legends and show
plt.xlabel("Evaluations")
plt.ylabel("RMSE")
plt.legend(loc="lower right", ncols=2, fontsize=7)
plt.xlim(0, 10_000)
plt.grid()
plt.tight_layout()
# plt.show()
plt.savefig("figures/dtlz_rmse_full.png")
