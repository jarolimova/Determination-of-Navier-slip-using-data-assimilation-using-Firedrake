"""module defining the forward model for the assimilation process in Firedrake.
This module includes functions for both steady and unsteady state simulations, handling the weak form of the
Navier-Stokes equations, and computing the analytic inlet profile for a given mesh and boundary conditions.
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import firedrake as fd
    import firedrake.adjoint as fda
from firedrake.__future__ import interpolate
from petsc4py import PETSc

from mpi4py import MPI
import ufl
from solver_settings import solver_parameters_steady, solver_parameters_shared
from timestep_iterator import TimestepIterator, RunningAverage

print = PETSc.Sys.Print

marks = {"wall": 1, "in": 2, "out": 3}

# Helper functions for vector field operations:


def u_n(v, n):
    """Compute the normal component of a vector field v at the facet normal n."""
    return fd.inner(v, n) * n


def u_t(v, n):
    """Compute the tangential component of a vector field v at the facet normal n."""
    return v - u_n(v, n)


def T(p, v, mu, dim=3):
    """Compute the stress tensor T for a given pressure p, velocity v, and dynamic viscosity mu."""
    identity = fd.Identity(dim)
    return -p * identity + 2 * mu * fd.sym(fd.grad(v))


def forward_steady(
    wall_control,
    u_in,
    mesh,
    w,
    mu,
    rho,
    slip_penalty,
    picard_weight=0.0,
    beta=1e3,
    stabilization="IP",
    stab_v=0.0,
    stab_p=0.0,
    stab_i=0.0,
):
    W = w.function_space()
    w0 = fd.Function(W)

    F = weak_form(
        wall_control,
        mesh,
        w,
        w0,
        mu,
        rho,
        slip_penalty,
        beta=beta,
        stabilization=stabilization,
        stab_v=stab_v,
        stab_p=stab_p,
        stab_i=stab_i,
        u_in_nitsche=u_in,
    )
    J_picard = fd.derivative(F, w)
    F_picard = ufl.replace(F, {w0: w})
    J_picard = ufl.replace(J_picard, {w0: w})
    F_newton = ufl.replace(F, {w0: w})
    J_newton = fd.derivative(F_newton, w)

    J = picard_weight * (J_picard - J_newton) + J_newton
    F = picard_weight * (F_picard - F_newton) + F_newton

    problem = fd.NonlinearVariationalProblem(F, w, bcs=[], J=J)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solver_parameters_steady
    )
    solver.solve()
    return w


def forward_unsteady(
    wall_control,
    u_in_list,
    mesh,
    w,
    mu,
    rho,
    timestepping,
    slip_penalty,
    picard_weight=0.0,
    beta=1e3,
    stabilization="IP",
    stab_v=0.0,
    stab_p=0.0,
    stab_i=0.0,
    data_dict=dict(),
    average=False,
    return_w=False,
):
    W = w.function_space()
    u_in = fd.Function(W.sub(0))
    w0 = fd.Function(W)
    w_prev = fd.Function(W)

    F = weak_form(
        wall_control,
        mesh,
        w,
        w0,
        mu,
        rho,
        slip_penalty,
        beta=beta,
        stabilization=stabilization,
        stab_v=stab_v,
        stab_p=stab_p,
        stab_i=stab_i,
        u_in_nitsche=u_in,
        dt=timestepping.dt,
        w_prev=w_prev,
    )
    J_picard = fd.derivative(F, w)
    F_picard = ufl.replace(F, {w0: w})
    J_picard = ufl.replace(J_picard, {w0: w})
    F_newton = ufl.replace(F, {w0: w})
    J_newton = fd.derivative(F_newton, w)

    J = picard_weight * (J_picard - J_newton) + J_newton
    F = picard_weight * (F_picard - F_newton) + F_newton

    problem = fd.NonlinearVariationalProblem(F, w, bcs=[], J=J)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solver_parameters_shared
    )

    vel, press = w.subfunctions
    vel0, press0 = w_prev.subfunctions
    timestamps = list(data_dict.keys())
    timestamps.sort()
    timestep_iterator = iter(TimestepIterator(timestamps))
    if timestep_iterator.max_interval < timestepping.dt:
        raise Warning(
            f"data have better temporal resolution than the simulation! (data_dt = {timestep_iterator.max_interval}, simulation_dt = {timestepping.dt})"
        )
    t = timestepping(-1)
    n = fd.FacetNormal(mesh)
    if return_w:
        w_list = []
        estimated_data = dict()
        w_list.append(w.copy(deepcopy=True))
    else:
        J_list = []
        t_list = []
    # find the interval including t
    t_next, t_lower, t_upper = next(timestep_iterator, (None, None, None))
    # FIRST TIME STEP
    while t_upper is not None and t_upper < t:
        t_next, t_lower, t_upper = next(timestep_iterator, (None, None, None))
    if average:
        # initialize avg oject and add vel to it if it is inside of the interval
        running_average_obj = RunningAverage(fd.Function(vel.function_space()))
        if t_lower is not None and t_upper is not None and t_lower <= t <= t_upper:
            running_average_obj(vel)
    # interpolate:
    elif t_next is not None and t_next == t:
        # evaluate if the initial time is the same as in the data
        if return_w:
            estimated_data[t_next] = fd.project(vel, vel.function_space())
        else:
            J_list.append(l2_norm_with_optional_interpolation(vel, data_dict[t_next]))
            t_list.append(t_next)
        # more to the next interval
        t_next, t_lower, t_upper = next(timestep_iterator, (None, None, None))
    # FOLLOWING TIME STEPS
    for i, uin in enumerate(u_in_list):
        u_in.assign(uin)
        t = timestepping(i)
        solver.solve()
        if return_w:
            w_list.append(w.copy(deepcopy=True))
        vel, press = w.subfunctions
        vel0, press0 = w_prev.subfunctions
        if average:
            # if we are in the current interval, update running average
            if t_lower is not None and t_upper is not None and t_lower <= t <= t_upper:
                running_average_obj(vel)
            # we entered the next interval
            elif t_upper is not None and t > t_upper:
                # evaluate
                if return_w:
                    estimated_data[t_next] = running_average_obj.current_average
                else:
                    J_list.append(
                        l2_norm_with_optional_interpolation(
                            running_average_obj.current_average, data_dict[t_next]
                        )
                    )
                    t_list.append(t_next)
                # move to the next interval
                t_next, t_lower, t_upper = next(timestep_iterator, (None, None, None))
                running_average_obj = RunningAverage(fd.Function(vel.function_space()))
                if (
                    t_lower is not None
                    and t_upper is not None
                    and t_lower <= t <= t_upper
                ):
                    running_average_obj(vel)

        # interpolate:
        elif t_next is not None and t_next <= t:
            # if we got to the next data point, evaluate the distance from data
            weight = (t - t_next) / timestepping.dt
            interpolated = fd.Function(vel.function_space())
            interpolated.assign((1 - weight) * vel + weight * vel0)
            if return_w:
                estimated_data[t_next] = fd.project(interpolated, vel.function_space())
            else:
                J_list.append(
                    l2_norm_with_optional_interpolation(interpolated, data_dict[t_next])
                )
                t_list.append(t_next)
            t_next, t_lower, t_upper = next(timestep_iterator, (None, None, None))
        w_prev.assign(w)

    # add the last average object to the list
    if average:
        if t_next is not None:
            if return_w:
                estimated_data[t_next] = running_average_obj.current_average
            else:
                J_list.append(
                    l2_norm_with_optional_interpolation(
                        running_average_obj.current_average, data_dict[t_next]
                    )
                )
                t_list.append(t_next)

    # return results
    if return_w:
        return w_list, estimated_data
    else:
        Jdata = 0.0
        for i in range(1, len(J_list)):
            Jdata += 0.5 * (J_list[i - 1] + J_list[i]) * (t_list[i] - t_list[i - 1])
        return Jdata


def l2_norm_with_optional_interpolation(v, data):
    if v.function_space() != data.function_space():
        print("interpolating to data space ....")
        interpolated = fd.assemble(
            interpolate(
                v,
                data.function_space(),
                allow_missing_dofs=True,
            )
        )
        return fd.assemble(
            fd.inner(
                interpolated - data,
                interpolated - data,
            )
            * fd.dx
        )
    else:
        print("leaving as is ....")
        return fd.assemble(
            fd.inner(
                v - data,
                v - data,
            )
            * fd.dx
        )


def weak_form(
    wall_control,
    mesh,
    w,
    w0,
    mu,
    rho,
    slip_penalty,
    beta=1e3,
    stokes=False,
    stabilization="IP",
    stab_v=0.0,
    stab_p=0.0,
    stab_i=0.0,
    u_in_nitsche=None,
    dt=None,
    w_prev=None,
    bndry_marks=marks,
):
    dim = mesh.topological_dimension()
    # Split trial and test functions
    u, p = fd.split(w)
    W = w.function_space()
    v, q = fd.TestFunctions(W)
    u0, p0 = fd.split(w0)

    # Compute minimum cell diameter for stabilization scaling
    fda.pause_annotation()
    edgelen_loc = fd.project(
        fd.CellDiameter(mesh), fd.FunctionSpace(mesh, "DG", 0)
    ).dat.data_ro.min()
    edgelen = mesh.mpi_comm().allreduce(edgelen_loc, MPI.MIN)
    fda.continue_annotation()

    n = fd.FacetNormal(mesh)

    # Stokes base
    F = (
        +fd.inner(T(p, u, mu, dim=dim), fd.grad(v)) * fd.dx
        + fd.inner(q, fd.div(u)) * fd.dx
    )
    # convective term
    if not stokes:
        F += rho * fd.inner(fd.grad(u) * u0, v) * fd.dx

    # Directional do-nothing boundary condition on the outlet
    F += (
        -rho
        * 0.5
        * fd.conditional(fd.gt(fd.inner(u, n), 0.0), 0.0, 1.0)
        * fd.inner(u, n)
        * fd.inner(fd.inner(u, n), fd.inner(v, n))
        * fd.ds(bndry_marks["out"])
    )

    # Navier slip boundary condition (tangential part) on the wall
    F += (
        slip_penalty.control_variable_to_penalty(wall_control)
        * fd.inner(u_t(u, n), u_t(v, n))
        * fd.ds(bndry_marks["wall"])
    )

    # Interior Penalty (IP) stabilization
    if stabilization == "IP":
        if stab_v > 0.0:
            F += (
                stab_v
                * rho
                * fd.avg(edgelen * edgelen)
                * fd.inner(fd.jump(fd.grad(u)), fd.jump(fd.grad(v)))
                * fd.dS
            )
        if stab_p > 0.0:
            F += (
                stab_p
                / rho
                * fd.avg(edgelen * edgelen)
                * fd.inner(fd.jump(fd.grad(p)), fd.jump(fd.grad(q)))
                * fd.dS
            )
        if stab_i > 0.0:
            F += (
                stab_i
                * rho
                * fd.avg(edgelen * edgelen)
                * pow(fd.inner(u("+"), n("+")), 2)
                * fd.inner(fd.jump(fd.grad(u)), fd.jump(fd.grad(v)))
                * fd.dS
            )
    # SUPG and pressure stabilization
    elif stabilization == "SUPG":
        # simple SUPG convection stabilization see https://doi.org/10.1016/S0065-2156(08)70153-4
        res_strong = rho * fd.grad(u) * u - fd.div(T(p, u, mu, dim=dim))
        vnorm2 = fd.dot(u, u)
        h = edgelen
        tau_SUPG = stab_v * ((60.0 * mu / h / h) ** 2 + 4.0 * vnorm2 / h**2) ** (-0.5)
        F_SUPG = (
            fd.conditional(fd.gt(rho, 0.0), 1.0, 0.0)
            * fd.cell_avg(tau_SUPG)
            * fd.inner(res_strong, fd.grad(v) * u)
            * fd.dx
        )
        # Local Projection stabilization for pressure
        F_S = (
            stab_p
            * (1.0 / mu)
            * fd.inner(p - fd.cell_avg(p), q - fd.cell_avg(q))
            * fd.dx
        )
        F += F_SUPG + F_S
    else:
        raise ValueError(f"{stabilization} is not a valid stabilization!")

    # Nitsche's method for enforcing u.n=0 on the wall (normal component)
    F += (
        -fd.inner(u_n(T(p, u, mu, dim=dim) * n, n), u_n(v, n))
        * fd.ds(bndry_marks["wall"])
        + fd.inner(u_n(u, n), u_n(T(q, v, mu, dim=dim) * n, n))
        * fd.ds(bndry_marks["wall"])
        + beta
        * mu
        / edgelen
        * fd.inner(u_n(u, n), u_n(v, n))
        * fd.ds(bndry_marks["wall"])
    )

    # Nitsche's method for enforcing u_t=0 on the outlet (tangential component)
    F += (
        -fd.inner(u_t(T(p, u, mu, dim=dim) * n, n), u_t(v, n))
        * fd.ds(bndry_marks["out"])
        + fd.inner(u_t(u, n), u_t(T(q, v, mu, dim=dim) * n, n))
        * fd.ds(bndry_marks["out"])
        + beta
        * mu
        / edgelen
        * fd.inner(u_t(u, n), u_t(v, n))
        * fd.ds(bndry_marks["out"])
    )

    # Nitsche's method for inflow boundary condition if provided
    if u_in_nitsche is not None:
        F += (
            -fd.inner(T(p, u, mu, dim=dim) * n, v) * fd.ds(bndry_marks["in"])
            + fd.inner(u - u_in_nitsche, T(q, v, mu, dim=dim) * n)
            * fd.ds(bndry_marks["in"])
            + beta
            * mu
            / edgelen
            * fd.inner(u - u_in_nitsche, v)
            * fd.ds(bndry_marks["in"])
        )

    # Add time derivative for unsteady problems
    if dt is not None and w_prev is not None:
        u_prev, p_prev = fd.split(w_prev)
        F += rho / dt * fd.inner(u - u_prev, v) * fd.dx

    return F
