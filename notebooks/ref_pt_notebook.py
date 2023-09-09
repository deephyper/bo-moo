import os
import numpy as np

dpath = "/lus/eagle/projects/datascience/regele/polaris/multi-objective-hpo/metric_data"
#dataset = "fashion_mnist"
#dataset = "cifar10"
dataset = "colorectal_histology"
section = "valid_set.pkl.gz"
data_path = os.path.join(dpath, dataset, section)
print(data_path)

import gzip

with gzip.GzipFile(data_path, "rb") as f:
    data = np.load(f, allow_pickle=True)

objective_columns = ["valid-acc", "latency", "size_MB"]

# Fixed constants
fixed_fidelities = {
    "Optimizer": "SGD",
    "N": 5.0, # Depth Multiplier (value in {1, 3, 5})
    "W": 16.0, # Width Multiplier (value in {4, 8, 16})
    "Resolution": 1.0, # Resolution Multiplier (value in {0.25, 0.5, 1.0})
    # "epoch": 200,
}

selection = None
for k,v in fixed_fidelities.items():
    if selection is None:
        selection = data["features"][k] == v
    else:
        selection = selection & (data["features"][k] == v)

sdata = data[selection]
print(f"{len(sdata) / len(data) * 100} %")

objs = sdata["labels"][objective_columns].values
objs[:,0] = 100 - objs[:,0] # Accuracy becomes Error Rate

from deephyper.skopt.moo import non_dominated_set

pf_mask = non_dominated_set(objs)

import matplotlib.pyplot as plt

labels = ["Error Rate", "Latency", "Size"]
idx = [(0, 1), (0, 2), (1, 2)]

for i, j in idx:
    plt.figure()
    plt.scatter(objs[~pf_mask][:, i], objs[~pf_mask][:, j])
    plt.scatter(objs[pf_mask][:, i], objs[pf_mask][:, j], color="red")
    plt.xlabel(labels[i])
    plt.ylabel(labels[j])

    if i == 0:
        plt.xlim(0, 15)
    plt.tight_layout()

    plt.savefig(f"{dataset}_{labels[i]}_vs_{labels[j]}.png")
