import pandas as pd
import numpy as np

import matplotlib as mpl

### Romain's style ###


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
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

width, height = set_size(252, fraction=1.0)

# width = 5
# height = width/1.618
fontsize = 9

mpl.rcParams.update({
    'font.size': fontsize,
    'figure.figsize': (width, height), 
    'figure.facecolor': 'white', 
    'savefig.dpi': 360, 
    'figure.subplot.bottom': 0.125, 
    'figure.edgecolor': 'white',
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize
})


### Begin reading data from results ###

import matplotlib.pyplot as plt
import pandas as pd

import deephyper_benchmark as dhb
dhb.load("JAHSBench")
from deephyper_benchmark.lib.jahsbench.metrics import PerformanceEvaluator
evaluator = PerformanceEvaluator()

PROBLEM = "jahs"

cmap = ["r", "g", "b", "c"]
nmap = ["Random", "D-MoBO", "NSGAII", "MoTPE"]

for NNODES in [10, 20, 30, 40]:
    for si, SEARCH in enumerate(['dh', 'random', 'nsgaii', 'tpe']):
        full_list = []
        times = np.asarray([t for t in range(600, 7201, 600)])
        for SEED in range(5):
            print(f"opening results/{PROBLEM}-{SEARCH}-{NNODES}-{SEED}/results.csv ...")
            df = pd.read_csv(f"results/{PROBLEM}-{SEARCH}-{NNODES}-{SEED}/results.csv", sep=',')
            try:
                t0 = df[df["job_id"] == 0]["m:timestamp_submit"].iloc[0]
                t_idx = "m:timestamp_gather"
            except KeyError:
                t0 = df[df["job_id"] == 0]["m:timestamp_start"].iloc[0]
                t_idx = "m:timestamp_end"
            curr_list = []
            for time in times:
                results = df[df[t_idx] <= t0 + time][["objective_0", "objective_1", "objective_2"]].to_numpy()
                curr_list.append(evaluator.hypervolume(results))
            full_list.append(curr_list)
        sorted_list = np.asarray(full_list)
        sorted_list.sort(axis=0)
        q1 = sorted_list[:2, :].mean(axis=0)
        q2 = sorted_list[2, :]
        q3 = sorted_list[3:, :].mean(axis=0)
        plt.plot(times / 60, q2, color=f"{cmap[si]}", label=f"{nmap[si]}")
        plt.fill_between(times / 60, q1, q3, color=f"{cmap[si]}", alpha=0.2)
    plt.legend(loc="lower right")
    plt.ylabel("Hypervolume")
    plt.xlabel("Time (mins)")
    plt.savefig(f"scaling-fmnist-{NNODES}.png")
    plt.clf()
