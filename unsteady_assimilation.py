import json
import os
import csv
import time
from datetime import datetime, timedelta
from math import ceil
from itertools import islice, cycle
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import firedrake as fd
    import firedrake.adjoint as fda
from petsc4py import PETSc

from MRI_tools.MRI_firedrake import MRI
from data_loading import load_to_mri, read_mesh_h5, read_data_h5, generate_mask
from firedrake.__future__ import interpolate
from optimization_setup import ErrorFunctionalUnsteady
from forward import forward_unsteady, marks
from input_args import get_args_unsteady, prepare_folder
import finite_elements
from startups import string_to_startup
from slip_penalty import SlipPenalty, NU, RHO, CHAR_LEN
from timestep_iterator import DataInterpolation, SimulationTimestepping

print = PETSc.Sys.Print

iteration = 0


def callback(j, m):
    if args.no_theta:
        theta = args.init_theta
        wc = slip_penalty.theta_to_control_variable(theta)
        wc_to_print = float(wc)
        u_ins = m
        wc_reg = 0.0

    elif args.no_vin:
        wc = m
        wc_to_print = float(wc)
        theta = slip_penalty.control_variable_to_theta(float(wc))
        u_ins = u_in_list
        wc_reg = 0.0
    else:
        wc = m[0]
        if args.wall_control_nonconst:
            wc_to_print = fd.assemble(wc * fd.ds(marks["wall"])) / area_wall
            theta = slip_penalty.control_variable_to_theta(wc_to_print)
            wc_reg = error_functional.grad_l2_wall_control_ds(wc)
        else:
            wc_to_print = float(wc)
            theta = slip_penalty.control_variable_to_theta(float(wc))
            wc_reg = 0.0
        u_ins = m[1:]
    if not args.skip_h5 and not args.no_vin:
        with fd.CheckpointFile(f"{folder}/uin_chpt.h5", "w") as h5_uins:
            h5_uins.save_mesh(mesh)
            for i, uin in enumerate(u_in_list):
                h5_uins.save_function(uin, idx=i)

    global iteration
    print(
        f"iter = {iteration}, J = {j}, wall_control = {wc_to_print}, theta = {theta}, grad_reg = {error_functional.grad_reg(u_ins)}, time_reg = {error_functional.time_reg(u_ins)}, timegrad_reg = {error_functional.timegrad_reg(u_ins)}, kinetic_energy_reg = {error_functional.kinetic_energy(u_ins)}, wc_reg = {wc_reg}"
    )
    if fd.COMM_WORLD.rank == 0:
        with open(output_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, float(theta), j])
    if args.picard == iteration:
        picard_weight.assign(0.0)
        print(f"Picard -> Newton (at the end of iteration {iteration})")
    iteration += 1


if __name__ == "__main__":
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Datetime: {now}")
    initial_time = time.time()
    args = get_args_unsteady()

    fda.continue_annotation()
    # read mesh
    mesh = read_mesh_h5(args.meshname, args.data_folder)

    # initialize finite element spaces
    element_class = finite_elements.string_to_element[args.element]
    FE = element_class(mesh)
    W = FE.W
    V = FE.V
    Q = FE.P
    R = fd.FunctionSpace(mesh, "R", 0)
    w = fd.Function(W, name="w")
    u, p = w.subfunctions
    area_wall = fd.assemble(fd.project(fd.Constant(1.0), Q) * fd.ds(marks["wall"]))

    if args.data_h5_mesh is not None:
        h5path = os.path.join(
            args.data_folder, args.data_h5_mesh, "p1p1", args.dataname
        )
        with open(h5path + ".json", "r") as js:
            data_timelist_h5 = json.load(js)
        # print(data_timelist_h5)
        nsteps = len(data_timelist_h5)
        mri_functions, h5_mesh = read_data_h5(
            h5path + ".h5", args.data_h5_mesh, nsteps=nsteps, printing=False
        )
        timesteps_list_h5 = [
            data_timelist_h5[i + 1] - data_timelist_h5[i]
            for i in range(len(data_timelist_h5) - 1)
        ]
        avg_timestep_h5 = sum(timesteps_list_h5) / len(timesteps_list_h5)
        data_period = round(nsteps * avg_timestep_h5, 12)
        if args.T_end is None:
            # if T_end not specified, select 1 data_period to be the T_end
            args.T_end = data_period
        T = args.T_end
        n_timesteps = ceil(T / avg_timestep_h5)
        ncycles = ceil(n_timesteps / nsteps)
        data_timelist = [
            round(
                c + i * (data_timelist_h5[-1] - data_timelist_h5[0] + avg_timestep_h5),
                12,
            )
            for i in range(ncycles)
            for c in data_timelist_h5
        ][:n_timesteps]
        print(data_timelist)

    else:
        fda.pause_annotation()
        # read mri
        mri = MRI.from_json(os.path.join(args.MRI_json_folder, args.dataname))
        padding = 2
        space_type = args.MRI_space
        hexahedral = args.MRI_space == "DG"
        if args.operator_interpolation:
            generate_mask(
                mri, mesh, padding=padding, space_type=space_type, hexahedral=hexahedral
            )
        mri_functions = load_to_mri(
            mri=mri,
            mesh=mesh,
            padding=padding,
            space_type=space_type,
            hexahedral=hexahedral,
        )
        fda.continue_annotation()
        data_period = round(len(mri_functions) * mri.timestep, 12)

        if args.T_end is None:
            # if T_end not specified, select 1 data_period to be the T_end
            args.T_end = data_period
        T = args.T_end
        n_timesteps = ceil(T / mri.timestep)
        data_timelist = [(i + 0.5) * mri.timestep for i in range(n_timesteps)]

    dt = args.dt
    t0 = -dt * args.presteps
    timestepping = SimulationTimestepping(t0=t0, dt=dt)
    if t0 < 0.0:
        startup = string_to_startup["linear"](start=t0, end=0.0)
    else:
        startup = string_to_startup["none"](start=t0, end=0.0)

    print(args)
    # prepare folder
    folder, output_file = prepare_folder(
        args, fd.COMM_WORLD, empty_folder=False if args.taylor_test else True
    )
    # init output csv
    if not args.taylor_test:
        with open(output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "theta", "J"])

    # save mri data
    pvddatafile = fd.output.VTKFile(f"{folder}/mri.pvd")
    for t, d in zip(data_timelist, mri_functions):
        pvddatafile.write(d, time=t)

    #  make data_timelist and data_list
    if args.operator_interpolation:
        # periodic list starting with 0
        data_list = list(islice(cycle(mri_functions), n_timesteps))
    else:
        # interpolate data
        interpolated_mri_functions = [
            fd.assemble(interpolate(mri_function, V)) for mri_function in mri_functions
        ]
        for fun in interpolated_mri_functions:
            fun.rename("data")
        # periodic list starting with 0
        data_list = list(islice(cycle(interpolated_mri_functions), n_timesteps))

    # save data for reference
    pvddatafile = fd.output.VTKFile(f"{folder}/data.pvd")
    data_dict = dict()
    for t, d in zip(data_timelist, data_list):
        pvddatafile.write(d, time=t)
        data_dict[t] = d

    mu = fd.Constant(NU * RHO)
    rho = fd.Constant(RHO)
    picard_weight = fd.Constant(1.0 if args.picard > 0 else 0.0)
    slip_penalty = SlipPenalty(
        control_variable=args.wall_control, gamma=args.gamma_star
    )
    # prepare wall_control (= transformed theta) as real value or function in space
    if args.wall_control_nonconst:
        wall_control = fd.Function(Q)
        wall_control.dat.data[:] = slip_penalty.theta_to_control_variable(
            args.init_theta
        )
    else:
        wall_control = fd.Function(R).assign(
            slip_penalty.theta_to_control_variable(args.init_theta)
        )

    # set initial guess for v_in
    u_in_list_len = ceil((T - t0) / dt) + 1
    if args.vin_path == "data":
        fda.pause_annotation()
        data_interpolation = DataInterpolation(data_dict)
        u_in_list = [
            fd.assemble(
                startup(timestepping(i))
                * interpolate(data_interpolation(timestepping(i)), V)
            )
            for i in range(u_in_list_len)
        ]
        fda.continue_annotation()
    elif args.vin_path is not None:
        u_in_list, _ = read_data_h5(
            args.vin_path + ".h5",
            args.meshname,
            dataname="v",
            nsteps=u_in_list_len,
            printing=False,
        )
    else:
        u_in_list = [fd.Function(V) for i in range(u_in_list_len)]
    uinstart_pvdfile = fd.output.VTKFile(f"{folder}/u_in_start.pvd")
    for i, uin in enumerate(u_in_list):
        uin.rename("v")
        uinstart_pvdfile.write(uin, time=timestepping(i))

    # run forward
    J_data = forward_unsteady(
        wall_control,
        u_in_list,
        mesh,
        w,
        mu,
        rho,
        timestepping,
        slip_penalty,
        picard_weight=picard_weight,
        stabilization=args.stabilization,
        stab_v=args.stab_v,
        stab_p=args.stab_p,
        stab_i=args.stab_i,
        data_dict=data_dict,
        average=args.average,
    )

    # setup error functional and compute it
    error_functional = ErrorFunctionalUnsteady(mesh, dt, T, CHAR_LEN)

    J_scaled = 1 / (2.0 * T * pow(CHAR_LEN, 3)) * J_data
    print("J = ", J_scaled)

    J_alpha = args.alpha * error_functional.grad_reg(u_in_list)
    print("R_alpha = ", J_alpha)
    J_gamma = args.gamma * error_functional.time_reg(u_in_list)
    print("R_gamma = ", J_gamma)
    J_delta = args.delta * error_functional.timegrad_reg(u_in_list)
    print("R_delta = ", J_delta)
    J_epsilon = args.epsilon * error_functional.kinetic_energy(u_in_list)
    print("R_epsilon = ", J_epsilon)
    Jreg = J_scaled + J_alpha + J_gamma + J_delta + J_epsilon

    if args.wall_control_nonconst:
        Jbeta = args.beta * error_functional.grad_l2_wall_control_ds(wall_control)
        print("R_beta = ", Jbeta)
        Jreg += Jbeta
    print("Jreg = ", Jreg)

    # setup controls (= mark variables which will able to change between the iterations)
    picard_control = fda.Control(picard_weight)

    if args.no_theta and args.no_vin:
        print("No control variables -> no optimization")
        exit()
    elif args.no_theta:
        m = [fda.Control(u_in) for u_in in u_in_list]
    elif args.no_vin:
        m = fda.Control(wall_control)
    else:
        m = [fda.Control(wall_control)]
        m += [fda.Control(u_in) for u_in in u_in_list]
    Jhat = fda.ReducedFunctional(Jreg, m, eval_cb_post=callback)

    fda.pause_annotation()
    # tape = fda.get_working_tape()
    # tape.visualise("tape.pdf")

    before_optimization_time = time.time()
    print(
        f"Setup time: {str(timedelta(seconds=round(before_optimization_time-initial_time, 3)))}"
    )
    if args.taylor_test:
        h_wc = fd.Function(wall_control.function_space())
        h_wc.dat.data[:] = 0.001
        h_uin_list = [fd.Function(u_in.function_space()) for u_in in u_in_list]
        for h_uin in h_uin_list:
            h_uin.dat.data[:] = 0.001
        conv_rate = fda.taylor_test(
            Jhat, [wall_control] + u_in_list, [h_wc] + h_uin_list
        )
        print("convergence rates: ", conv_rate)
        exit()

    # start optimization
    if args.optimization_method == "IPOPT":
        opt_problem = fda.MinimizationProblem(Jhat)
        params = {
            "print_user_options": "yes",
            "linear_solver": "mumps",  # "MA57",
            "tol": 0.2,
            "dual_inf_tol": args.gtol,
            "acceptable_obj_change_tol": args.ftol,
            "max_iter": 100,
            "print_level": 5 * (fd.COMM_WORLD.rank == 0),
            # "hessian_approximation": "limited-memory",
        }
        opt_solver = fda.IPOPTSolver(opt_problem, parameters=params)
        m_opt = opt_solver.solve()
    else:
        opts = {"disp": fd.COMM_WORLD.rank == 0}
        if args.optimization_method == "L-BFGS-B":
            opts["ftol"] = args.ftol
            opts["gtol"] = args.gtol
        elif args.optimization_method == "CG":
            opts["gtol"] = args.gtol
        elif args.optimization_method == "Newton-CG":
            opts["xtol"] = args.ftol
        m_opt = fda.minimize(
            Jhat,
            method=args.optimization_method,
            options=opts,
        )
    after_optimization_time = time.time()

    # split m_opt and save results
    if args.no_theta:
        wall_control_opt = wall_control
        u_in_opt = m_opt
    else:
        if args.no_vin:
            wall_control_opt = m_opt
            u_in_opt = u_in_list
        else:
            wall_control_opt = m_opt[0]
            u_in_opt = m_opt[1:]
        if args.wall_control_nonconst:
            fd.output.VTKFile(f"{folder}/wall_control.pvd").write(wall_control_opt)
        else:
            theta_opt = slip_penalty.control_variable_to_theta(float(wall_control_opt))
            if fd.COMM_WORLD.rank == 0:
                print(f"Optimal theta: {float(theta_opt)}")

    if not args.skip_h5:
        with fd.CheckpointFile(f"{folder}/u_in_opt.h5", "w") as h5file:
            h5file.save_mesh(mesh)
            for i, u in enumerate(u_in_opt):
                u.rename("v")
                h5file.save_function(u, idx=i)

    # compute optimal velocity, pressure and save
    w_opt, estimated_data = forward_unsteady(
        wall_control_opt,
        u_in_opt,
        mesh,
        w,
        mu,
        rho,
        timestepping,
        slip_penalty,
        picard_weight=picard_weight,
        stabilization=args.stabilization,
        stab_v=args.stab_v,
        stab_p=args.stab_p,
        stab_i=args.stab_i,
        data_dict=data_dict,
        average=args.average,
        return_w=True,
    )
    if not args.skip_h5:
        with fd.CheckpointFile(
            f"{folder}/ns_opt.h5", "w"
        ) as vh5file, fd.CheckpointFile(f"{folder}/p_opt.h5", "w") as ph5file:
            vh5file.save_mesh(mesh)
            ph5file.save_mesh(mesh)
            for i, result in enumerate(w_opt):
                vel, press = result.subfunctions
                vel.rename("v")
                press.rename("p")
                vh5file.save_function(vel, idx=i)
                ph5file.save_function(press, idx=i)

    pvdnsoptfile = fd.output.VTKFile(f"{folder}/ns_opt.pvd")
    for i, result in enumerate(w_opt):
        vel, press = result.subfunctions
        vel.rename("v")
        press.rename("p")
        pvdnsoptfile.write(vel, press, time=timestepping(i - 1))

    pvdnsoptfile_est = fd.output.VTKFile(f"{folder}/ns_opt_data_est.pvd")
    for t in estimated_data.keys():
        est_data = estimated_data[t]
        projected = fd.project(est_data, vel.function_space())
        projected.rename("v")
        pvdnsoptfile_est.write(projected, time=t)

    end_time = time.time()

    print(
        f"Setup time: {str(timedelta(seconds=round(before_optimization_time-initial_time, 3)))}"
    )
    print(
        f"Optimization time: {str(timedelta(seconds=round(after_optimization_time-before_optimization_time, 3)))}"
    )
    print(
        f"Finalization time: {str(timedelta(seconds=round(end_time-after_optimization_time, 3)))}"
    )
    print(50 * "=")
    print(f"Total time: {str(timedelta(seconds=round(end_time-initial_time, 3)))}")
