#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence produces only rudimentary progress messages and does not provide
batch distribution or timing prints, as `example_experiment2.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

>>> import cma  # doctest:+SKIP
>>> def fmin(fun, x0):
...     return cma.fmin(fun, x0, 2, {'verbose':-9})

"""
from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from TuRBO.turbo import Turbo1
import numpy as np
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser

def fmin(problem, x0, verbose):
    dim = len(x0)
    fmin_ = Turbo1(problem, 
                    lb = -5 * np.ones(dim), 
                    ub = 10 * np.ones(dim),
                    n_init = 20,
                    max_evals = 100,
                    verbose = False,
                    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
                    min_cuda=1024,  # Run on the CPU for small datasets
                    device="cpu",  # "cpu" or "cuda"
                    dtype="float64")
    return fmin_.optimize()

### input
suite_name = "bbob"
output_folder = "scipy-optimize-fmin"
# fmin = Turbo1
budget_multiplier = 1  # increase to 10, 100, ...

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

### go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    x0 = problem.initial_solution
    # apply restarts while neither the problem is solved nor the budget is exhausted
    while (problem.evaluations < problem.dimension * budget_multiplier
           and not problem.final_target_hit):
        fmin(problem, x0, verbose=False)  # here we assume that `fmin` evaluates the final/returned solution
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                    (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

