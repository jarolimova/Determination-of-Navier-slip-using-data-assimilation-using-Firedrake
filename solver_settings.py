""" Solver settings for Firedrake simulations, including steady and unsteady solvers.
"""

solver_parameters_shared = {
    "snes_converged_reason": "",
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_linesearch_type": "nleqerr",  # cp, bt
    "snes_atol": 1e-10,
    "snes_rtol": 1e-10,
    "snes_max_it": 100,
    "mat_mumps_icntl_24": 1,  # detect null pivots
    "mat_mumps_icntl_14": 300,  # work array, multiple to estimate to allocate
    "mat_mumps_cntl_1": 1e-2,  # relative pivoting threshold
}

solver_parameters_steady = {
    "snes_monitor": "",
}
solver_parameters_steady.update(solver_parameters_shared)
