"""This script generates artificial unsteady data for a fluid flow simulation using Firedrake.
"""

import os
import json
import argparse
from math import ceil, floor
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import firedrake as fd
from petsc4py import PETSc
from startups import string_to_startup
import velocity_factors
from forward import forward_unsteady, AnalyticInletProfile
import finite_elements
from slip_penalty import SlipPenalty
from timestep_iterator import SimulationTimestepping

print = PETSc.Sys.Print


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("meshname", type=str, help="name of the mesh to be used")
    parser.add_argument(
        "theta",
        type=float,
        help="theta value (Navier slip parameter) from interval [0, 1]",
    )
    parser.add_argument(
        "--element",
        type=str,
        default="mini",
        help="finite element to be used",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="theta",
        help="name of data case (they can be different in velocity, rho, nu and amount of noise)",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=0.8,
        help="velocity of the flow (default: 0.8 m/s)",
    )
    parser.add_argument(
        "--velocity_factor",
        type=str,
        default="const",
        help="shape of velocity magnitude in time, other options: pulse",
    )
    parser.add_argument(
        "--startup",
        type=str,
        default="none",
        help="startup for interval [0, 0.5]",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="slip weight gamma: theta/gamma(1-theta)",
    )
    parser.add_argument(
        "--rho", type=float, default=1050.0, help="fluid density (default: 1050 kg/m^3)"
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=3.71e-6,
        help="kinematic viscosity (default: 3.71e-6 m^2/s)",
    )
    parser.add_argument(
        "--stab_v",
        type=float,
        default=None,
        help="weight for velocity part of IP stab, default: 0.05*stab_i for p1p1, 0.0 for all other elements",
    )
    parser.add_argument(
        "--stab_p",
        type=float,
        default=None,
        help="weight for pressure part of IP stab, default: 0.1 for p1p1 element and 0.0 for all other elements",
    )
    parser.add_argument(
        "--stab_i",
        type=float,
        default=None,
        help="weight for normal part of velocity w.r.t. element edges in IP stab,  default: 0.01 for p1p1, 0.0 for all other elements",
    )
    parser.add_argument(
        "--meshpath",
        type=str,
        default=None,
        help="the folder containing the mesh folder",
    )
    parser.add_argument(
        "--average_interval",
        type=float,
        default=None,
        help="save result with with fewer timestamps averaged over time",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=0.8,
        help="length of the simulation, default: 0.8",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="length of the simulation timestep, default: 0.01",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="location of the data folder (where the artificial data will be saved), default: data",
    )
    args = parser.parse_args()
    return args


def generate_artificial_unsteady_data(
    theta: float,
    meshpath: str,
    element: str = "p1p1",
    stab_v=None,
    stab_p=None,
    stab_i=None,
    data_folder: str = "data",
    name: str = "theta",
    v_mag: float = 0.8,
    dt: float = 0.01,
    T: float = 0.8,
    avg_interval=None,
    nu: float = 3.71e-6,
    rho: float = 1050.0,
    gamma: float = 0.25,
    startup_name: str = "none",
    velocity_factor_name: str = "const",
    plot_velocity_factor: bool = True,
):
    """Generate artificial unsteady data for a fluid flow simulation using Firedrake."""
    meshname = os.path.basename(meshpath)
    folder = os.path.join(data_folder, meshname, element)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(meshpath + "_cuts.json", "r") as js:
        bnd_data = json.load(js)
    mesh = fd.Mesh(meshpath + ".msh", name=meshname)
    element_class = finite_elements.string_to_element[element]
    FE = element_class(mesh)
    W = FE.W
    V = FE.V
    if FE.stable:
        if stab_v is None:
            stab_v = 0.0
        if stab_p is None:
            stab_p = 0.0
        if stab_i is None:
            stab_i = 0.0
    else:
        if stab_i is None:
            stab_i = 0.01
        if stab_v is None:
            stab_v = 0.05 * stab_i
        if stab_p is None:
            stab_p = 1.0

    w = fd.Function(W)
    mu = fd.Constant(rho * nu)
    slip_penalty = SlipPenalty(control_variable="theta", gamma=gamma)
    analytic_inlet_profile = AnalyticInletProfile(
        bnd_data["in"], mesh, mu, slip_penalty
    )
    startup = string_to_startup[startup_name](start=0.0, end=0.5)
    if velocity_factor_name == "const":
        velocity_factor = velocity_factors.ConstVelocityFactor(velocity_average=v_mag)
    elif velocity_factor_name == "pulse":
        velocity_factor = velocity_factors.PulseVelocityFactor(v_max=v_mag, period=T)
    if plot_velocity_factor:
        velocity_factors.plot_velocity_factor(
            f"{folder}/velocity_factor_{velocity_factor_name}", velocity_factor, end=T
        )
    u_in_list = []
    timestepping = SimulationTimestepping(t0=0.0, dt=dt)
    nsteps = ceil(T / dt)
    u_in_list = []
    for i in range(nsteps):
        ufl_in = analytic_inlet_profile(
            theta,
            fd.Constant(startup(timestepping(i)) * velocity_factor(timestepping(i))),
        )
        u_in_list.append(fd.project(fd.as_vector(ufl_in), V))

    avg_timestamps = floor(T / avg_interval)
    endpoints = [i * avg_interval for i in range(avg_timestamps + 1)]
    avg_ts = [
        round(0.5 * (endpoints[i] + endpoints[i + 1]), 12)
        for i in range(avg_timestamps)
    ]
    data_dict = dict()
    for ts in avg_ts:
        data_dict[ts] = None
    print(data_dict.keys())

    result_list, estimated_data = forward_unsteady(
        fd.Constant(theta),
        u_in_list,
        mesh,
        w,
        mu,
        fd.Constant(rho),
        timestepping,
        slip_penalty,
        picard_weight=0.0,
        stab_v=stab_v,
        stab_p=stab_p,
        stab_i=stab_i,
        data_dict=data_dict,
        average=True,
        return_w=True,
    )
    print(estimated_data.keys())

    results_name = f"{name}{theta}_timedep"
    print(f"saving results to {folder}/{results_name} ...")
    timesteps_list = []
    with fd.CheckpointFile(
        f"{folder}/{results_name}.h5", "w"
    ) as h5file, fd.CheckpointFile(
        f"{folder}/{results_name}_pressure.h5", "w"
    ) as ph5file:
        h5file.save_mesh(mesh)
        ph5file.save_mesh(mesh)
        for i, result in enumerate(result_list):
            t = timestepping(i - 1)
            vel, press = result.subfunctions
            h5file.save_function(vel, idx=i, name="data")
            ph5file.save_function(press, idx=i, name="p")
            timesteps_list.append(t)
    with open(f"{folder}/{results_name}.json", "w") as json_file:
        json.dump(timesteps_list, json_file)

    if avg_interval is not None:
        print(f"averaging over intervals of length {avg_interval} ...")
        avg_results_name = f"{name}{theta}_avg{avg_interval}_timedep"
        with fd.CheckpointFile(f"{folder}/{avg_results_name}.h5", "w") as avg_h5file:
            avg_h5file.save_mesh(mesh)
            for i, t in enumerate(estimated_data.keys()):
                est_data = estimated_data[t]
                projected = fd.project(est_data, vel.function_space())
                projected.rename("v")
                avg_h5file.save_function(projected, idx=i, name="data")
        with open(f"{folder}/{avg_results_name}.json", "w") as json_file:
            json.dump(list(estimated_data.keys()), json_file)
    return


if __name__ == "__main__":
    args = get_args()
    if args.meshpath is None:
        raise ValueError(
            "Please provide the path to the mesh folder using --meshpath argument."
        )
    meshpath = os.path.join(args.meshpath, args.meshname, args.meshname)
    generate_artificial_unsteady_data(
        args.theta,
        meshpath,
        element=args.element,
        stab_v=args.stab_v,
        stab_p=args.stab_p,
        stab_i=args.stab_i,
        name=args.name,
        v_mag=args.velocity,
        dt=args.dt,
        T=args.T,
        avg_interval=args.average_interval,
        nu=args.nu,
        rho=args.rho,
        gamma=args.gamma,
        startup_name=args.startup,
        velocity_factor_name=args.velocity_factor,
        data_folder=args.data_folder,
    )
