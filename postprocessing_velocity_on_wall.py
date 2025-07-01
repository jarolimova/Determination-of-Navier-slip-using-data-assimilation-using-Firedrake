import os
from math import ceil
import argparse
import firedrake as fd
from data_loading import read_data_h5
from forward import marks
from matplotlib import pyplot as plt

plt.rcParams["text.usetex"]
plt.rcParams.update({"font.size": 13})


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot pressure drop and average velocity"
    )
    parser.add_argument(
        "meshname",
        type=str,
        help="name of the mesh to plot",
    )
    parser.add_argument(
        "dataname",
        type=str,
        help="name of the data to plot",
    )
    parser.add_argument(
        "--numerical_setting",
        type=str,
        default="p1p1_stab0.0005_1.0_0.01",
        help="numerical setting for the simulation, default: p1p1_stab0.0005_1.0_0.01",
    )
    parser.add_argument(
        "--initializations",
        type=str,
        default="init_theta0.7_data",
        help="initialization setting, default: init_theta0.7_data",
    )
    parser.add_argument(
        "--discretization",
        type=str,
        default="facet_0.25_T0.8_dt0.01_pr2",
        help="discretization setting, default: dt0.01_T0.8_pr2",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        default="alpha0.001_gamma0.001_eps0.001_avg",
        help="regularization setting, default: alpha0.001_gamma0.001_eps0.001_avg",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="picard0",
        help="solver setting, default: picard0",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="location of the results folder which are being plotted, default: results",
    )
    parser.add_argument(
        "--plot_folder",
        type=str,
        default="plots",
        help="location of the folder where plots are going to be saved, default: plots",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.plot_folder):
        os.makedirs(args.plot_folder, exist_ok=True)
    folder_path = os.path.join(
        args.results_folder,
        args.meshname,
        args.dataname,
        args.numerical_setting,
        args.initializations,
        args.discretization,
        args.regularization,
        args.solver,
    )
    T = None
    dt = None
    for text_piece in args.discretization.split("_"):
        if "T" in text_piece:
            T = float(text_piece.replace("T", ""))
        if "dt" in text_piece:
            dt = float(text_piece.replace("dt", ""))
    if T is None or dt is None:
        raise ValueError("T and/or dt values not found in the path!")
    print(T, dt)
    nsteps = ceil(T / dt) + 1
    v_list, mesh = read_data_h5(
        os.path.join(folder_path, "ns_opt.h5"), args.meshname, "v", nsteps
    )
    Q = fd.FunctionSpace(mesh, "CG", 1)
    volume = fd.assemble(fd.project(fd.Constant(1.0), Q) * fd.dx)
    area_wall = fd.assemble(fd.project(fd.Constant(1.0), Q) * fd.ds(marks["wall"]))
    t_list = []
    wall_integrals = []
    volume_integrals = []
    relative = []
    for i, v in enumerate(v_list):
        v.rename("v")
        t_list.append(i * dt)
        volume_int = (fd.assemble(fd.sqrt(fd.inner(v, v)) * fd.dx)) / volume
        wall_int = (
            fd.assemble(fd.sqrt(fd.inner(v, v)) * fd.ds(marks["wall"]))
        ) / area_wall
        volume_integrals.append(volume_int)
        wall_integrals.append(wall_int)
        relative.append(wall_int / volume_int)
    # maximal flow
    max_vol = max(volume_integrals)
    # index of max flow
    idx = volume_integrals.index(max_vol)
    print("max_wall_L1: ", wall_integrals[idx])
    volume_integrated = 0
    wall_integrated = 0
    for i in range(1, len(t_list)):
        volume_integrated += (
            0.5
            * (volume_integrals[i - 1] + volume_integrals[i])
            * (t_list[i] - t_list[i - 1])
        )
        wall_integrated += (
            0.5
            * (wall_integrals[i - 1] + wall_integrals[i])
            * (t_list[i] - t_list[i - 1])
        )
    print(
        "percentage of integral: ",
        100 * wall_integrated / volume_integrated,
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_list, volume_integrals, label=r"domain = $\Omega$")
    ax.plot(t_list, wall_integrals, label=r"domain = $\Gamma_{wall}$")
    ax.grid()
    ax.set_xlim(xmin=0.0)
    ax.set_xlim(xmax=T)
    ax.legend(fontsize=20)
    ax.set_xlabel("time (s)", fontsize=20)
    ax.set_ylabel(r"$||v||_{L^1}$", fontsize=20)
    for format in ["png", "pdf"]:
        fig.savefig(
            os.path.join(args.plot_folder, f"v_on_wall_{args.meshname}.{format}"),
            bbox_inches="tight",
        )
    fig.clf()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_list, relative)
    ax.grid()
    ax.set_xlim(xmin=0.0)
    ax.set_xlim(xmax=T)
    ax.legend(fontsize=20)
    ax.set_xlabel("time (s)", fontsize=20)
    ax.set_ylabel(
        r"$\frac{||v||_{L^1(\Gamma_{wall})}|\Omega|}{||v||_{L^1(\Omega)}|\Gamma_{wall}|}$",
        fontsize=20,
    )
    for format in ["png", "pdf"]:
        fig.savefig(
            os.path.join(args.plot_folder, f"relative_{args.meshname}.{format}"),
            bbox_inches="tight",
        )
    fig.clf()
