# Setup info-level logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - " + \
           "%(message)s",
    force=True,
)

# Problem parameters
BB_BUDGET = 200 # 200 eval budget
NDIMS = 8 # 8 vars
NOBJS = 3 # 3 objs

# Set DTLZ problem environment variables
import os
os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = str(NDIMS)
os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = str(NOBJS)
os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = "2" # DTLZ2 problem
os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.6" # [x_o, .., x_d]*=0.6

# Load DTLZ benchmark suite, nothing to install
import deephyper_benchmark as dhb
dhb.load("DTLZ")


# Necessary IF statement otherwise it will enter in a infinite loop
# when loading the 'run' function from a subprocess
if __name__ == "__main__":
    import numpy as np
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    # Run HPO-pipeline with default configuration of hyperparameters
    from deephyper_benchmark.lib.dtlz import hpo
    from deephyper.evaluator import RunningJob, Evaluator
    config = hpo.problem.default_configuration
    print(config)
    res = hpo.run(RunningJob(parameters=config))
    print(f"{res=}")

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        hpo.run,
        method="process",
        method_kwargs={
            "num_workers": 4,
        },
    )

    # define your search and execute it
    search = CBO(hpo.problem, evaluator,
                 moo_scalarization_strategy="Chebyshev")

    # solve with BB_BUDGET evals
    results = search.search(max_evals=BB_BUDGET)
    print(results)

    # gather performance stats
    from deephyper.skopt.moo import pareto_front, hypervolume
    from ast import literal_eval
    obj_vals = np.asarray([literal_eval(fi)
                           for fi in results["objective"].values])
    hv_vals = []
    bbf_num = []
    for i in range(10, BB_BUDGET + 1, 10):
        hv_vals.append(hypervolume(obj_vals[:i, :], 4.0 * np.ones(NOBJS)) /
                       (64.0 - np.pi / 3.0))
        bbf_num.append(i)

    # plot performance over time
    from matplotlib import pyplot as plt
    plt.scatter(bbf_num, hv_vals, c='r', s=50, label="")
    plt.plot(bbf_num, hv_vals, 'k--')
    plt.xlabel("Number of blackbox function evaluations")
    plt.ylabel("Percent of total hypervolume dominated")
    plt.legend()
    plt.tight_layout()
    plt.show()
