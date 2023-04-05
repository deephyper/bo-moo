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

    # define the search method and scalarization
    search = CBO(hpo.problem, evaluator,
                 moo_scalarization_strategy="rChebyshev")

    # solve with BB_BUDGET evals
    results = search.search(max_evals=BB_BUDGET)

    # gather performance stats
    from ast import literal_eval
    from deephyper_benchmark.lib.dtlz.metrics import PerformanceEvaluator

    # Extract objective values from dataframe
    obj_vals = np.asarray([literal_eval(fi)
                           for fi in results["objective"].values])
    # Initialize performance arrays
    rmse_vals = []
    npts_vals = []
    hv_vals = []
    bbf_num = []
    # Create a performance evaluator for this problem and loop over budgets
    perf_eval = PerformanceEvaluator()
    for i in range(10, BB_BUDGET, 10):
        hv_vals.append(perf_eval.hypervolume(obj_vals[:i, :]))
        rmse_vals.append(perf_eval.rmse(obj_vals[:i, :]))
        npts_vals.append(perf_eval.numPts(obj_vals[:i, :]))
        bbf_num.append(i)
    # Don't forget final budget
    hv_vals.append(perf_eval.hypervolume(obj_vals))
    rmse_vals.append(perf_eval.rmse(obj_vals))
    npts_vals.append(perf_eval.numPts(obj_vals))
    bbf_num.append(BB_BUDGET)

    # dump results to a csv file
    import csv

    with open(FILENAME, "a") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Problem name", "metric", "input vars", "objectives"]
                        + bbf_num)
        writer.writerow([f"DLTZ{PROB_NUM}", "hypervol", str(NDIMS), str(NOBJS)]
                         + hv_vals)
        writer.writerow([f"DLTZ{PROB_NUM}", "RMSE", str(NDIMS), str(NOBJS)]
                         + rmse_vals)
        writer.writerow([f"DLTZ{PROB_NUM}", "num pts", str(NDIMS), str(NOBJS)]
                         + npts_vals)

    ## plot performance over time
    #from matplotlib import pyplot as plt
    #plt.scatter(bbf_num, hv_vals, c='r', s=50, label="")
    #plt.plot(bbf_num, hv_vals, 'k--')
    #plt.xlabel("Number of blackbox function evaluations")
    #plt.ylabel("Percent of total hypervolume dominated")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
