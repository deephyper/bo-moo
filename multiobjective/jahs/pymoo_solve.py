import csv
import numpy as np
import os
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
import sys

from jahs_bench.api import Benchmark
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

class JahsBench(ElementwiseProblem):

    def __init__(self, **kwargs):
        variables = {
            "LearningRate": Real(bounds=(1.0e-3, 1.0)),
            "WeightDecay": Real(bounds=(1.0e-5, 1.0e-3)),
            "Activation": Choice(options=["ReLU", "Hard", "Mish"]),
            "TrivialAugment": Binary(),
            "Op1": Integer(bounds=(0, 4)),
            "Op2": Integer(bounds=(0, 4)),
            "Op3": Integer(bounds=(0, 4)),
            "Op4": Integer(bounds=(0, 4)),
            "Op5": Integer(bounds=(0, 4)),
            "Op6": Integer(bounds=(0, 4)),
        }
        super().__init__(vars=variables, n_obj=2, n_ieq_constr=0, **kwargs)

        ### JAHS bench settings ###
        DATASET = "cifar10"
        MODEL_PATH = "."
        self.NEPOCHS = 200
        # Define the benchmark
        self.benchmark = Benchmark(
                task=DATASET,
                save_dir=MODEL_PATH,
                kind="surrogate",
                download=True
        )
        self.count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """ ParMOO compatible simulation function wrapping jahs-bench.
    
        Args:
            x (numpy structured array): Configuration array with same keys
                as jahs-bench.
    
        Returns:
            numpy.ndarray: 1D array of size 2 containing the validation loss
            (100 - accuracy) and latency.
    
        """
    
        sx = []
        # Default config
        config = {
            'Optimizer': 'SGD',
            'N': 5,
            'W': 16,
            'Resolution': 1.0,
        }
        # Update config using X
        for key in X.keys():
            config[key] = X[key]
        # Pymoo truncates Hardswish to Hard
        if config['Activation'] == "Hard":
            config['Activation'] = "Hardswish"
        # Evaluate and return
        result = self.benchmark(config, nepochs=self.NEPOCHS)
        sx.append(100 - result[self.NEPOCHS]['valid-acc'])
        sx.append(result[self.NEPOCHS]['latency'])
        out["F"] = sx.copy()
        self.count += 1
        print(f"finished evaluation {self.count}: {sx}")
        # Dump results to csv file
        if self.count == 1:
            with open(FILENAME, "w") as fp:
                writer = csv.writer(fp)
                writer.writerow(sx)
        else:
            with open(FILENAME, "a") as fp:
                writer = csv.writer(fp)
                writer.writerow(sx)


# Set the random seed from CL or system clock
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
else:
    from datetime import datetime
    SEED = int(datetime.now().timestamp())
FILENAME = f"pymoo-cifar/results_seed{SEED}.csv"


# Solve DTLZ2 problem w/ NSGA-II (pop size 100) in pymoo
algorithm = NSGA2(pop_size=20,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  #survival=RankAndCrowdingSurvival()
                  )
res = minimize(JahsBench(),
               algorithm,
               ("n_gen", 50),
               save_history=True,
               seed=SEED,
               verbose=False)
#
## Extract all objective values
#obj_vals = []
#for row in res.history:
#    for fi in row.result().F:
#        obj_vals.append(fi)
#obj_vals = np.asarray(obj_vals)
#
## Dump results to csv file
#with open(FILENAME, "w") as fp:
#    writer = csv.writer(fp)
#    for row in obj_vals:
#        writer.writerow(row)
