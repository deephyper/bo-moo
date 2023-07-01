
import numpy as np
from parmoo.extras.libe import libE_MOOP
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint, FixedWeights
from parmoo.surrogates import LocalGaussRBF
from parmoo.optimizers import TR_LBFGSB

from parmoo.objectives.obj_lib import single_sim_out


### Problem dimensions ###
num_des = 8
num_obj = 2

### Budget variables ###
n_search_sz = 200 # 200 pt initial DOE
n_per_batch = 40 # batch size = 40
iters_limit = 20 # run for 20 iterations


def sim_func(x):
    """ ParMOO compatible simulation function wrapping jahs-bench.

    Args:
        x (numpy structured array): Configuration array with same keys
            as jahs-bench.

    Returns:
        numpy.ndarray: 1D array of size 2 containing the validation loss
        (100 - accuracy) and latency.

    """

    # Evaluate and return
    sx = np.zeros(2)
    sx[0] = 100 - x['LearningRate'] * 1.0e3
    sx[1] = x['LearningRate'] * 1.0e3
    return sx


if __name__ == "__main__":
    """ Start solving problem with TR solver """

    moop_tr = libE_MOOP(TR_LBFGSB)
    # 2 continuous variables
    moop_tr.addDesign({'name': "LearningRate",
                        'des_type': "continuous",
                        'lb': 1.0e-3, 'ub': 1.0, 'des_tol': 1.0e-3})
    moop_tr.addDesign({'name': "WeightDecay",
                        'des_type': "continuous",
                        'lb': 1.0e-5, 'ub': 1.0e-3, 'des_tol': 1.0e-5})
    # 2 categorical variables
    moop_tr.addDesign({'name': "Activation",
                        'des_type': "categorical",
                        'levels': ["ReLU", "Hardswish", "Mish"]})
    moop_tr.addDesign({'name': "TrivialAugment",
                        'des_type': "categorical",
                        'levels': ["on", "off"]})
    # 6 integer variables
    for i in range(1, 7):
        moop_tr.addDesign({'name': f"Op{i}",
                            'des_type': "integer",
                            'lb': 0, 'ub': 4})
    # JAHS benchmark
    moop_tr.addSimulation({'name': "jahs",
                            'm': num_obj,
                            'sim_func': sim_func,
                            'search': LatinHypercube,
                            'surrogate': LocalGaussRBF,
                            'hyperparams': {'search_budget': n_search_sz}})
    # Minimize 2 objectives
    moop_tr.addObjective({'name': "valid-loss",
                           'obj_func': single_sim_out(moop_tr.getDesignType(),
                                                      moop_tr.getSimulationType(),
                                                      ("jahs", 0))})
    moop_tr.addObjective({'name': "latency",
                           'obj_func': single_sim_out(moop_tr.getDesignType(),
                                                      moop_tr.getSimulationType(),
                                                      ("jahs", 1))})
    # q acquisition functions, 1 fixed
    weights = np.ones(2)
    weights[1] = 1.0e2
    moop_tr.addAcquisition({'acquisition': FixedWeights,
                            'hyperparams': {'weights': weights}})
    for i in range(n_per_batch-1):
        moop_tr.addAcquisition({'acquisition': RandomConstraint,
                                'hyperparams': {}})
    # Solve and dump to csv
    moop_tr.solve(sim_max=100)
    results_tr = moop_tr.getObjectiveData()


