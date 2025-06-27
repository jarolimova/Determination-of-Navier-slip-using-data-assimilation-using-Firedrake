"""module for handling command line arguments and folder structure for the assimilation process in Firedrake
"""

import argparse
import os
from shutil import rmtree
from pathlib import Path


def get_args_steady():
    """Get command line arguments for steady state assimilation."""
    parser = get_args()
    parser.add_argument(
        "--timestep",
        type=int,
        default=0,
        help="timestep of the data to assimilate, default: 0",
    )
    parser.add_argument(
        "--wall_control",
        type=str,
        default="theta",
        help="default: theta, options: penalty (theta/(gamma*(1-theta))), logarithm (log(penalty))",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="the regularization weight of grad of velocity at the inlet, default: 0.0",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="the regularization weight for distance from analytical profile, default: 0.0",
    )
    args = parser.parse_args()
    return args


def get_args_unsteady():
    """Get command line arguments for unsteady state assimilation."""
    parser = get_args()
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="the regularization weight of grad of velocity at the inlet, default: 0.0",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="the regularization weight for time derivative of the inlet, default: 0.0",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="the regularization weight for time derivative of the gradient velocity at the inlet, default: 0.0",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="the regularization weight for kinetic energy at the inlet, default: 0.0",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="simulation timestep size",
    )
    parser.add_argument(
        "--T_end",
        type=float,
        default=None,
        help="end time for the simulation, default: last time step included in the data",
    )
    parser.add_argument(
        "--presteps",
        type=int,
        default=0,
        help="number of simulation steps to add at the start",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="average over intervals in error functional",
    )
    parser.add_argument(
        "--wall_control",
        type=str,
        default="logarithm",
        help="default: theta, options: penalty (theta/(gamma*(1-theta))), logarithm (log(penalty)), sliplength (mu/penalty)",
    )
    parser.add_argument(
        "--vin_path",
        type=str,
        default=None,
        help="path to the h5 file to be used as intial guess for v_in, has to have the same timestepping!",
    )
    args = parser.parse_args()
    return args


def get_args():
    """Get command line arguments for the assimilation process - common for both steady and unsteady state."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("meshname", type=str, help="name of the mesh to be used")
    parser.add_argument("dataname", type=str, help="name of the data")
    parser.add_argument(
        "--init_theta",
        type=float,
        default=0.75,
        help="theta initial value, default: 0.75",
    )
    parser.add_argument(
        "--no_theta", action="store_true", help="turn off theta as a control variable"
    )
    parser.add_argument(
        "--no_vin", action="store_true", help="turn off v_in as a control variable"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="the regularization weight for grad wall_control (makes sense only from --wall_control_nonconst), default: 0.0",
    )
    parser.add_argument(
        "--element",
        type=str,
        default="p1p1",
        help="finite element to be used, default: p1p1, other options: mini, th",
    )
    parser.add_argument(
        "--MRI_space",
        type=str,
        default="CG",
        help="MRI space - CG1 or DG0, default: CG, other option: DG",
    )
    parser.add_argument(
        "--operator_interpolation",
        action="store_true",
        help="include interpolation in the measurement operator",
    )
    parser.add_argument(
        "--data_h5_mesh",
        type=str,
        default=None,
        help="read data from h5 + json instead of npy + json using the provided meshname",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="location of the data folder, default: data",
    )
    parser.add_argument(
        "--MRI_json_folder",
        type=str,
        default="MRI_npy",
        help="location of the MRI_json_folder, default: MRI_npy",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="location of the results folder, default: results",
    )
    parser.add_argument(
        "--stabilization",
        type=str,
        default="IP",
        help="type of stabilization to use. default: IP, other options: SUPG",
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
        "--picard",
        type=int,
        default=0,
        help="for the given number of iterations use picard instead of newton",
    )
    parser.add_argument(
        "--gamma_star",
        type=float,
        default=0.25,
        help="slip weight gamma, default: 0.25",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-6,
        help="tolerance for termination of the minimization, f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-5,
        help="tolerance for termination of the minimization, max{|proj g_i | i = 1, ..., n} <= gtol",
    )
    parser.add_argument(
        "--optimization_method",
        type=str,
        default="L-BFGS-B",
        help="minimization method, default: L-BFGS-B, other options: TNC, SLSQP (box constraints), CG, Newton-CG, BFGS (without constraints)",
    )
    parser.add_argument(
        "--wall_control_nonconst",
        action="store_true",
        help="change from constant wall control to a function in space",
    )
    parser.add_argument(
        "--skip_h5",
        action="store_true",
        help="skip saving checkpoints and results to h5",
    )
    parser.add_argument(
        "--taylor_test",
        action="store_true",
        help="run taylor test instead of the optimization",
    )
    return parser


def prepare_folder(args, comm, empty_folder=True):
    """Prepare the folder structure for saving results based on the provided arguments.

    Args:
        args: Parsed command line arguments.
        comm: MPI communicator for parallel execution.
        empty_folder (bool): If True, empty the folder before saving results.
    """
    results_firedrake = args.results_folder
    element_foldername = args.element
    if args.stabilization == "IP":
        if args.stab_i is None:
            args.stab_i = 0.01 if args.element == "p1p1" else 0.0
        if args.stab_v is None:
            args.stab_v = 0.05 * args.stab_i
        if args.stab_p is None:
            args.stab_p = 1.0 if args.element == "p1p1" else 0.0
        element_foldername += f"_stab{args.stab_v}_{args.stab_p}_{args.stab_i}"
    elif args.stabilization == "SUPG":
        element_foldername += "_supg"
        if args.stab_v is None:
            args.stab_v = 1.0
        if args.stab_p is None:
            args.stab_p = 1.0
        if args.stab_v != 1.0 or args.stab_p != 1.0:
            element_foldername += f"_stab{args.stab_v}_{args.stab_p}"
    reg_foldername = f"alpha{args.alpha}"
    if args.__contains__("beta") and args.beta != 0.0:
        reg_foldername += f"_beta{args.beta}"
    reg_foldername += f"_gamma{args.gamma}"
    if args.__contains__("delta") and args.delta != 0.0:
        reg_foldername += f"_delta{args.delta}"
    if args.__contains__("epsilon") and args.epsilon != 0.0:
        reg_foldername += f"_eps{args.epsilon}"
    if args.__contains__("average") and args.average:
        reg_foldername += "_avg"
    if args.operator_interpolation:
        reg_foldername += "interp"
    if args.data_h5_mesh is not None:
        reg_foldername += "h5"
    inits_foldername = f"init_theta{args.init_theta}"
    if args.no_theta:
        inits_foldername += "_notheta"
    if args.no_vin:
        inits_foldername += "_novin"
    if args.__contains__("vin_path"):
        if args.vin_path == "data":
            inits_foldername += "_data"
        elif "checkpoint" in args.vin_path:
            inits_foldername += "_rest"
        elif args.vin_path is not None:
            inits_foldername += "_vinit"
    if args.__contains__("timestep") and args.wall_control != "theta":
        inits_foldername += f"_{args.wall_control}"
    else:
        if args.wall_control != "logarithm":
            inits_foldername += f"_{args.wall_control}"
    if args.__contains__("wall_control_nonconst") and args.wall_control_nonconst:
        inits_foldername += "_nonconst"
    discretization_foldername = f"facet_{args.gamma_star}"
    if args.__contains__("T_end") and args.__contains__("dt"):
        discretization_foldername += f"_T{args.T_end}_dt{args.dt}"
    if args.__contains__("presteps") and args.presteps > 0:
        discretization_foldername += f"_pr{args.presteps}"
    solver_foldername = f"picard{args.picard}"
    if args.optimization_method != "L-BFGS-B":
        solver_foldername += f"_{args.optimization_method}"
    dataname_foldername = f"{args.dataname}_{args.MRI_space}"
    if args.__contains__("timestep"):
        dataname_foldername += f"_ts{args.timestep}"
    folder = os.path.join(
        results_firedrake,
        args.meshname,
        dataname_foldername,
        element_foldername,
        inits_foldername,
        discretization_foldername,
        reg_foldername,
        solver_foldername,
    )
    path_len = len(folder)
    if path_len > 230:
        print(
            f"FOLDER PATH MIGHT BE TOO LONG FOR SAVING H5 FILES (limit around 245) - currently {path_len}"
        )
    output_file = f"{folder}/output.csv"
    if empty_folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        if comm.rank == 0:
            print(args)
            print("RESULTS FOLDER: ", folder)

        for f in os.listdir(folder):
            pth = os.path.join(folder, f)
            if comm.rank == 0:
                try:
                    os.remove(pth)
                except IsADirectoryError:
                    rmtree(pth)
    return folder, output_file
