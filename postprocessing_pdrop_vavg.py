from math import ceil
import os
import sys
import csv
import json
import argparse
import firedrake as fd
from firedrake.__future__ import interpolate
from data_loading import read_data_h5
from forward import marks
from matplotlib import pyplot as plt
from MRI_tools.MRI_firedrake import MRI
from data_loading import load_to_mri, read_mesh_h5


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot pressure drop and average velocity"
    )
    parser.add_argument(
        "--meshes",
        type=str,
        nargs="+",
        help="list of mesh names to plot",
    )
    parser.add_argument(
        "--datanames",
        type=str,
        nargs="+",
        help="list of data names to plot",
    )
    parser.add_argument(
        "--regularizations",
        type=str,
        nargs="+",
        default=["alpha0.001_gamma0.001_eps0.001_avg"],
        help="list of regularizations to plot, default: alpha0.001_gamma0.001_eps0.001_avg",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        nargs="+",
        default=["picard0"],
        help="list of solvers to plot, default: picard0",
    )
    parser.add_argument(
        "--initializations",
        type=str,
        nargs="+",
        default=["init_theta0.7_data"],
        help="list of initializations to plot, default: init_theta0.7_data",
    )
    parser.add_argument(
        "--numerical_setting",
        type=str,
        default="p1p1_stab0.0005_1.0_0.01",
        help="numerical setting for the simulation, default: p1p1_stab0.0005_1.0_0.01",
    )
    parser.add_argument(
        "--presteps",
        type=int,
        nargs="+",
        default=[2],
        help="list of presteps to show in the plot, default: [2]",
    )
    parser.add_argument(
        "--MRI_data",
        type=str,
        default=None,
        help="path to the corresponding MRI data in JSON+npy format to appear in the plot, default: None",
    )
    parser.add_argument(
        "--ground_truth_h5",
        type=str,
        default=None,
        help="path to the ground truth data in h5 format, default: None",
    )
    parser.add_argument(
        "--csv_save",
        action="store_true",
        help="if set, saves the data to a csv file, default: False",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="labels for the plot, can be 'operators', 'slip', 'presteps' or None, default: None",
    )
    parser.add_argument(
        "--figname_tail",
        type=str,
        default="",
        help="tail to be added to the figure name, default: ''",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="location of the results folder which are being plotted, default: results",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="location of the data folder containing meshes, default: data",
    )
    parser.add_argument(
        "--plot_folder",
        type=str,
        default="plots",
        help="location of the folder where plots are going to be saved, default: plots",
    )
    args = parser.parse_args()
    return args


def pressure_drop(p_list, mesh):
    """Calculate the pressure drop across the inlet and outlet of the mesh."""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    in_area = fd.assemble(fd.project(fd.Constant(1.0), Q) * fd.ds(marks["in"]))
    out_area = fd.assemble(fd.project(fd.Constant(1.0), Q) * fd.ds(marks["out"]))
    in_p_list = [fd.assemble(p * fd.ds(marks["in"])) / in_area for p in p_list]
    out_p_list = [fd.assemble(p * fd.ds(marks["out"])) / out_area for p in p_list]
    p_drop_list = [in_p - out_p for in_p, out_p in zip(in_p_list, out_p_list)]
    return p_drop_list


def average_velocity(v_list, mesh):
    """Calculate the average velocity across the mesh."""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    volume = fd.assemble(fd.project(fd.Constant(1.0), Q) * fd.dx)
    v_avg_list = [fd.assemble(fd.sqrt(fd.inner(v, v)) * fd.dx) / volume for v in v_list]
    return v_avg_list


def make_graphs(
    meshes,
    datanames,
    regs,
    solvers=["picard0"],
    inits=["init_theta0.7_data"],
    element="p1p1_stab0.0005_1.0_0.01",
    results_folder="results",
    show_presteps=[2],
    original_data=None,
    csv_save=False,
    ground_truth_h5=None,
    figname_tail="",
    labels=None,
):
    """Create graphs of pressure drop and average velocity for the given meshes and data."""
    if original_data is not None:
        mri = MRI.from_json(os.path.join(original_data))
    if ground_truth_h5 is not None:
        with open(ground_truth_h5 + ".json", "r") as js:
            data_timelist = json.load(js)
        nsteps = len(data_timelist)
        ground_truth_meshname = ground_truth_h5.split("/")[-3]
        ground_truth_data_list, ground_truth_mesh = read_data_h5(
            ground_truth_h5 + ".h5", ground_truth_meshname, nsteps=nsteps
        )
        ground_truth_pressure_list, ground_truth_pressure_mesh = read_data_h5(
            ground_truth_h5 + "_pressure.h5",
            ground_truth_meshname,
            dataname="p",
            nsteps=nsteps,
        )
    for meshname in meshes:
        if original_data is not None:
            mesh = read_mesh_h5(meshname, data_folder=args.data_folder)
            mri_functions = load_to_mri(
                mri=mri,
                mesh=mesh,
                padding=2,
                space_type="CG",
                hexahedral=False,
            )
            V = fd.VectorFunctionSpace(mesh, "CG", 1)
            interpolated_mri_functions = [
                fd.assemble(interpolate(mri_function, V))
                for mri_function in mri_functions
            ]
            mri_list = average_velocity(interpolated_mri_functions, mesh)
            mri_timesteps = [(i + 0.5) * mri.timestep for i in range(len(mri_list))]

        if ground_truth_h5 is not None:
            mesh = read_mesh_h5(meshname, data_folder=args.data_folder)
            V = fd.VectorFunctionSpace(mesh, "CG", 1)
            P = fd.FunctionSpace(mesh, "CG", 1)
            interpolated_ground_truth_data_list = [
                fd.assemble(interpolate(dat, V)) for dat in ground_truth_data_list
            ]
            interpolated_ground_truth_pressure_list = [
                fd.assemble(interpolate(dat, P)) for dat in ground_truth_pressure_list
            ]
            ground_truth_v_avg_list = average_velocity(
                interpolated_ground_truth_data_list, mesh
            )
            ground_truth_pressure_drop_list = pressure_drop(
                interpolated_ground_truth_pressure_list, ground_truth_pressure_mesh
            )

        discretizations = []
        for dataname in datanames:
            for init in inits:
                path = os.path.join(results_folder, meshname, dataname, element, init)
                if os.path.exists(path):
                    discretizations += os.listdir(path)
        discretizations = list(set(discretizations))
        cases = list(set(["_".join(disc.split("_")[:4]) for disc in discretizations]))
        print(cases)
        for case in cases:
            T = None
            dt = None
            for text_piece in case.split("_"):
                if "T" in text_piece:
                    T = float(text_piece.replace("T", ""))
                if "dt" in text_piece:
                    dt = float(text_piece.replace("dt", ""))
            if T is None or dt is None:
                raise ValueError("T and/or dt values not found in the path!")
            nsteps = ceil(T / dt) + 1
            case_disc = [disc for disc in discretizations if case in disc]
            case_disc.sort()
            fig, ((axp, axv)) = plt.subplots(2, 1, sharex=True)
            axp.set_xlabel("time (s)", fontsize=12)
            axv.set_xlabel("time (s)", fontsize=12)
            axp.set_ylabel("pressure drop", fontsize=12)
            axv.set_ylabel("average velocity", fontsize=12)
            axp.set_xlim(xmin=0.0)
            axv.set_xlim(xmin=0.0)
            axp.set_xlim(xmax=T)
            axv.set_xlim(xmax=T)
            # axp.set_ylim(ymin=-700)
            # axp.set_ylim(ymax=1500)
            # axv.set_ylim(ymin=0.0)
            # axv.set_ylim(ymax=0.9)
            save_plot = False
            if original_data is not None:
                axv.plot(
                    mri_timesteps,
                    mri_list,
                    color="k",
                    marker=".",
                    linestyle="",
                    label="MRI data",
                )
                save_plot = True
            if ground_truth_h5 is not None:
                axv.plot(
                    data_timelist,
                    ground_truth_v_avg_list,
                    color="k",
                    marker="",
                    linestyle="-",
                    label="ground truth",
                )
                axp.plot(
                    data_timelist,
                    ground_truth_pressure_drop_list,
                    color="k",
                    marker="",
                    linestyle="-",
                    label="ground truth",
                )

            for dataname in datanames:
                for init in inits:
                    for reg in regs:
                        for disc in case_disc:
                            for solver in solvers:
                                ppath = os.path.join(
                                    results_folder,
                                    meshname,
                                    dataname,
                                    element,
                                    init,
                                    disc,
                                    reg,
                                    solver,
                                    "p_opt.h5",
                                )
                                vpath = os.path.join(
                                    results_folder,
                                    meshname,
                                    dataname,
                                    element,
                                    init,
                                    disc,
                                    reg,
                                    solver,
                                    "ns_opt.h5",
                                )
                                label = dataname + "_" + reg
                                presteps = 0
                                if len(solver.split("_")) > 1:
                                    label += ", " + solver.split("_")[1]
                                if "notheta" in init:
                                    label += ", " + init.split("_")[1]

                                for text_piece in disc.split("_"):
                                    if "pr" in text_piece:
                                        label += ", " + text_piece
                                        presteps = int(text_piece.replace("pr", ""))
                                if labels is not None:
                                    if labels == "operators":
                                        if "_avg" in label:
                                            label = r"$\mathcal{T}_{avg}$"
                                        else:
                                            label = r"$\mathcal{T}_{interp}$"
                                    elif labels == "slip":
                                        if "notheta" in init:
                                            theta = init.split("_")[1].split("theta")[1]
                                        else:
                                            output_file = os.path.join(
                                                results_folder,
                                                meshname,
                                                dataname,
                                                element,
                                                init,
                                                disc,
                                                reg,
                                                solver,
                                                "output.csv",
                                            )
                                            with open(
                                                output_file, newline=""
                                            ) as csvfile:
                                                reader = csv.DictReader(csvfile)
                                                rows = list(reader)
                                                if rows:
                                                    last_row = rows[-1]
                                                    theta = round(
                                                        float(last_row["theta"]), 3
                                                    )
                                                else:
                                                    print("CSV file is empty.")
                                                    break
                                        label = r"$\theta=" + f"{theta}" + r"$"
                                    elif labels == "presteps":
                                        presteps = 0
                                        if "pr" in text_piece:
                                            label += ", " + text_piece
                                            presteps = int(text_piece.replace("pr", ""))
                                        t0 = -presteps * dt
                                        label = r"$t_0=" + f"{t0}" + r"\,$s"
                                    else:
                                        label = ""
                                if presteps in show_presteps:
                                    t_list = [
                                        (i - presteps) * dt
                                        for i in range(nsteps + presteps)
                                    ]
                                    p_drop_list = None
                                    v_avg_list = None
                                    if os.path.isfile(ppath):
                                        p_list, mesh = read_data_h5(
                                            ppath, meshname, "p", nsteps + presteps
                                        )
                                        p_drop_list = pressure_drop(p_list, mesh)
                                        axp.plot(
                                            t_list,
                                            p_drop_list,
                                            label=label,
                                        )
                                        save_plot = True
                                    if os.path.isfile(vpath):
                                        v_list, mesh = read_data_h5(
                                            vpath, meshname, "v", nsteps + presteps
                                        )
                                        v_avg_list = average_velocity(v_list, mesh)
                                        axv.plot(
                                            t_list,
                                            v_avg_list,
                                            label=label,
                                        )
                                        save_plot = True
                                    if (
                                        csv_save
                                        and p_drop_list is not None
                                        and v_avg_list is not None
                                    ):
                                        save_csv(
                                            os.path.join(
                                                args.plot_folder,
                                                f"{meshname}_{case}_{dataname}_{init}_{reg}_{disc}_{solver}.csv",
                                            ),
                                            t_list,
                                            p_drop_list,
                                            v_avg_list,
                                        )
            axp.grid()
            axv.grid()
            axp.legend(fontsize=12)
            axv.legend(fontsize=12)
            if save_plot:
                figname = f"pressure_drop_{meshname}_{case}{figname_tail}"
                for format in ["png", "pdf"]:
                    fig.savefig(
                        os.path.join(
                            args.plot_folder,
                            figname + f".{format}",
                        ),
                        bbox_inches="tight",
                    )


def save_csv(output_file, timesteps, pdrops, vavgs):
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "pdrop", "vavg"])
        for time, pdrop, vavg in zip(timesteps, pdrops, vavgs):
            writer.writerow([time, pdrop, vavg])
    return


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.plot_folder):
        os.makedirs(args.plot_folder, exist_ok=True)
    make_graphs(
        meshes=args.meshes,
        datanames=args.datanames,
        regs=args.regularizations,
        solvers=args.solvers,
        inits=args.initializations,
        element=args.numerical_setting,
        results_folder=args.results_folder,
        show_presteps=args.presteps,
        original_data=args.MRI_data,
        csv_save=args.csv_save,
        ground_truth_h5=args.ground_truth_h5,
        figname_tail=args.figname_tail,
        labels=args.labels,
    )
