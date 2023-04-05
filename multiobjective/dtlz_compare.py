# Setup info-level logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - " + \
           "%(message)s",
    force=True,
)


# Problem parameters
FILENAME = "dtlz_perf_results.csv"
PROB_NUM = "2"
BB_BUDGET = 200 # 200 eval budget
NDIMS = 8 # 8 vars
NOBJS = 3 # 3 objs

# Set up the problem only once from MAIN
if __name__ == "__main__":

    import os
    import sys

    # If present, read problem definition from CL
    if len(sys.argv) > 1:
        assert sys.argv[1] in [str(i) for i in range(1, 8)]
        PROB_NUM = sys.argv[1]
    if len(sys.argv) > 2:
        assert int(sys.argv[2]) > 0
        NDIMS = int(sys.argv[2])
    if len(sys.argv) > 3:
        assert int(sys.argv[3]) > 0 and int(sys.argv[3]) < NDIMS
        NOBJS = int(sys.argv[3])
    if len(sys.argv) > 4:
        assert int(sys.argv[4]) > 0
        BB_BUDGET = int(sys.argv[4])

    # Set DTLZ problem environment variables
    os.environ["DEEPHYPER_BENCHMARK_NDIMS"] = str(NDIMS)
    os.environ["DEEPHYPER_BENCHMARK_NOBJS"] = str(NOBJS)
    os.environ["DEEPHYPER_BENCHMARK_DTLZ_PROB"] = PROB_NUM # DTLZ2 problem
    os.environ["DEEPHYPER_BENCHMARK_DTLZ_OFFSET"] = "0.6" # [x_o, .., x_d]*=0.6

# Load DTLZ benchmark suite, nothing to install
import deephyper_benchmark as dhb
dhb.load("DTLZ")


# Necessary IF statement otherwise it will enter in a infinite loop
# when loading the 'run' function from a subprocess
if __name__ == "__main__":
    import numpy as np
    from deephyper.evaluator import RunningJob, Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from deephyper_benchmark.lib.dtlz import hpo

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        hpo.run,
        method="process",
        method_kwargs={
            "num_workers": 4,
        },
    )

    # define the search method(s) and scalarization
    search1 = CBO(hpo.problem, evaluator,
                  moo_scalarization_strategy="Chebyshev",
                  moo_rescale_weight=True,
                  moo_scalarization_weight="uniform sample")
    search2 = CBO(hpo.problem, evaluator,
                  moo_scalarization_strategy="Chebyshev",
                  moo_rescale_weight=False,
                  moo_scalarization_weight="uniform sample")

    # solve with BB_BUDGET evals
    results_scaled = search1.search(max_evals=BB_BUDGET)
    results_unscaled = search2.search(max_evals=BB_BUDGET)

    # gather performance stats
    from ast import literal_eval
    from deephyper_benchmark.lib.dtlz.metrics import PerformanceEvaluator

    # Extract objective values from dataframe
    obj_pts_scaled = np.asarray([literal_eval(fi) for
                                 fi in results_scaled["objective"].values])
    obj_pts_unscaled = np.asarray([literal_eval(fi) for
                                   fi in results_unscaled["objective"].values])
    # Initialize performance arrays
    hv_vals_scaled = []
    hv_vals_unscaled = []
    bbf_num = []
    # Create a performance evaluator for this problem and loop over budgets
    perf_eval = PerformanceEvaluator()
    for i in range(10, BB_BUDGET, 10):
        hv_vals_scaled.append(perf_eval.hypervolume(obj_pts_scaled[:i, :]))
        hv_vals_unscaled.append(perf_eval.hypervolume(obj_pts_unscaled[:i, :]))
        bbf_num.append(i)
    # Don't forget final budget
    hv_vals_scaled.append(perf_eval.hypervolume(obj_pts_scaled))
    hv_vals_unscaled.append(perf_eval.hypervolume(obj_pts_unscaled))
    bbf_num.append(BB_BUDGET)

    # plot performance over time
    from matplotlib import pyplot as plt
    plt.scatter(bbf_num, hv_vals_scaled, c='r', s=50, label="Hypervolumes for scaled solver")
    plt.scatter(bbf_num, hv_vals_unscaled, c='b', s=50, label="Hypervolumes for unscaled solver")
    plt.plot(bbf_num, hv_vals_scaled, 'r--')
    plt.plot(bbf_num, hv_vals_unscaled, 'b--')
    plt.xlabel("Number of blackbox function evaluations")
    plt.ylabel("Percent of total hypervolume dominated")
    plt.legend()
    plt.tight_layout()
    plt.show()
