import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

class TestProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        variables = {
            "x1": Real(bounds=(0.0, 1.0)),
            "c1": Choice(options=["good", "med", "bad"]),
            "b1": Binary(),
            "i1": Integer(bounds=(0, 4)),
        }
        super().__init__(vars=variables, n_obj=2, n_ieq_constr=0, **kwargs)
        self.count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        sx = np.zeros(2)
        base = 0.0
        if X['c1'] == "med":
            base += 1
        elif X['c1'] == "bad":
            base += 2
        if X['b1']:
            base += 1
        base += np.abs(X['i1'] - 2)
        sx[0] = base + X['x1']**2
        sx[1] = base + (1.0 - X['x1'])**2
        out["F"] = sx.copy()
        self.count += 1
        print(f"finished evaluation {self.count}: {sx}")

# Fix random seed
SEED = 0

# Solve w/ NSGA-II with MixedVariable settings in pymoo (10 gens with pop size 10 = 100 evals)
algorithm = NSGA2(pop_size=10,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  #survival=RankAndCrowdingSurvival()
                  )
res = minimize(TestProblem(),
               algorithm,
               ("n_gen", 10),
               save_history=True,
               seed=SEED,
               verbose=False)

# Extract all objective values
obj_vals = []
for row in res.history:
    for fi in row.result().F:
        obj_vals.append(fi)
obj_vals = np.asarray(obj_vals)

# Dump results
print(f"Len history: {len(res.history)} gens")
print(f"Num evals in history: {len(obj_vals)}")
