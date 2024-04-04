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
import pandas as pd

from deephyper.analysis import rank

CONF_BOUND = True

# Dirs, names, and colors
dirs = [
    "dtlz_mpi_logs-C_id",
    "dtlz_mpi_logs-C_mml",
    "dtlz_mpi_logs-C_qu",
    "dtlz_mpi_logs-L_id",
    "dtlz_mpi_logs-L_mml",
    "dtlz_mpi_logs-L_qu",
    "dtlz_mpi_logs-P_id",
    "dtlz_mpi_logs-P_mml",
    "dtlz_mpi_logs-P_qu",
    "dtlz_mpi_logs-Optuna",
    "parmoo-tr",
]
labels = [
    "CH-Id",
    "CH-MML",
    "CH-QU",
    "L-Id",
    "L-MML",
    "L-QU",
    "PBI-Id",
    "PBI-MML",
    "PBI-QU",
    "NSGAII",
    "TR",
]

colors = [
    "orange",
    "orange",
    "orange",
    "seagreen",
    "seagreen",
    "seagreen",
    "royalblue",
    "royalblue",
    "royalblue",
    "crimson",
    "pink",
]

linestyle = [
    "-",
    ":",
    "--",
    "-",
    ":",
    "--",
    "-",
    ":",
    "--",
    "-",
    "-",
]

# Gather performance stats
df = []
for di, DNAME in enumerate(dirs):
    for iseed in range(10):
        FNAME = f"{DNAME}/results_seed{iseed}.csv"
        try:
            with open(FNAME, "r") as fp:
                csv_reader = csv.reader(fp)
                bbf_num = [float(x) for x in csv_reader.__next__()]
                hv_vals = [float(x) for x in csv_reader.__next__()]
                gd_vals = [float(x) for x in csv_reader.__next__()]

            if len(bbf_num) != 100:
                from scipy.interpolate import interp1d

                f_hv = interp1d(bbf_num, hv_vals)
                f_gd = interp1d(bbf_num, gd_vals)
                bbf_num_new = np.linspace(100, 10_000, 100)
                hv_vals = f_hv(bbf_num_new)
                gd_vals = f_gd(bbf_num_new)
                bbf_num = bbf_num_new.tolist()

            rdf = pd.DataFrame(
                {"bbf_num": bbf_num, "hv_vals": hv_vals, "gd_vals": gd_vals}
            )
            rdf["seed"] = iseed
            rdf["exp"] = DNAME
            df.append(rdf)
        except FileNotFoundError:
            print(f"skipping {DNAME}")
df = pd.concat(df, axis=0)

# Rankings
task_bbf_num = []
task_scores = []
task_rankings = []
for _, group_df in df.groupby(["seed"]):
    group_labels = []
    group_bbf_num = []
    group_gd = []
    for gv, gdf in group_df.groupby(["exp"]):
        # group_labels.append("-".join(gv))
        group_labels.append(gv)
        group_bbf_num.append(gdf["bbf_num"].values)
        group_gd.append(gdf["gd_vals"].values)

    group_bbf_num = np.array(group_bbf_num)
    group_gd = np.array(group_gd)

    ranks = np.zeros_like(group_gd).astype(int)
    for i in range(group_gd.shape[1]):
        r = rank(group_gd[:, i], decimals=5) + 1
        ranks[:, i] = r

    task_bbf_num.append(group_bbf_num)
    task_scores.append(group_gd)
    task_rankings.append(ranks)

task_bbf_num = np.array(task_bbf_num).astype(float)
task_scores = np.array(task_scores).astype(float)
task_rankings = np.array(task_rankings).astype(float)

conf = 1.96
n = task_rankings.shape[0]

average_bbf_num = np.mean(task_bbf_num, axis=0)

average_rankings = np.mean(task_rankings, axis=0)
stde_rankings = conf * np.std(task_rankings, axis=0) / np.sqrt(n)

average_scores = np.mean(task_scores, axis=0)
stde_scores = conf * np.std(task_scores, axis=0) / np.sqrt(n)

fig = plt.figure()

for di, DNAME in enumerate(dirs):
    i = group_labels.index(DNAME)

    if n > 0:
        # Plot mean
        plt.plot(
            average_bbf_num[i],
            average_rankings[i],
            linestyle=linestyle[di],
            color=f"{colors[di]}",
            label=labels[di],
        )
        # If more than 1 result, plot std devs
        if n > 1 and CONF_BOUND:
            plt.fill_between(
                average_bbf_num[i],
                average_rankings[i] - stde_rankings[i],
                average_rankings[i] + stde_rankings[i],
                color=f"{colors[di]}",
                alpha=0.2,
            )

# Add legends and show
plt.xlabel("Evaluations")
plt.ylabel("Ranking")
plt.xlim(0, 10_000)
plt.ylim(0)
plt.grid()
plt.tight_layout()
fig.legend(ncols=1, bbox_to_anchor=(1.13, 0.94), fontsize=6)
plt.savefig("figures/dtlz1_rank_from_gd.png", bbox_inches="tight")
# plt.show()

fig = plt.figure()

for di, DNAME in enumerate(dirs):
    i = group_labels.index(DNAME)

    if n > 0:
        # Plot mean
        plt.plot(
            average_bbf_num[i],
            average_scores[i],
            linestyle=linestyle[di],
            color=f"{colors[di]}",
            label=labels[di],
        )
        # If more than 1 result, plot std devs
        if n > 1 and CONF_BOUND:
            plt.fill_between(
                average_bbf_num[i],
                average_scores[i] - stde_scores[i],
                average_scores[i] + stde_scores[i],
                color=f"{colors[di]}",
                alpha=0.2,
            )

# Add legends and show
plt.xlabel("Evaluations")
plt.ylabel("GD+")
plt.xlim(0, 10_000)
plt.ylim(0)
plt.grid()
plt.tight_layout()
fig.legend(ncols=1, bbox_to_anchor=(1.13, 0.94), fontsize=6)
plt.savefig("figures/dtlz1_gd.png", bbox_inches="tight")
# plt.show()
