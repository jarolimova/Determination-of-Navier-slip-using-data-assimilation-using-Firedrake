import os
from math import ceil
import firedrake as fd
from data_loading import read_data_h5
from forward import marks
from matplotlib import pyplot as plt

plt.rcParams["text.usetex"]
plt.rcParams.update({"font.size": 13})

if __name__ == "__main__":
    results_folder = "results"
    element_inits = "p1p1_stab0.0005_1.0_0.01/init_theta0.7_data"
    # element_inits = "p1p1_stab0.0005_1.0_0.01/init_theta0.98_notheta_data"
    reg_solver = "alpha0.001_gamma0.001_eps0.001_avg/picard0"

    volunteers = ["vol01", "vol03", "vol10", "vol13", "vol17", "vol19", "vol20"]
    for vol in volunteers:
        meshes = [
            f"long_015_{vol}",
            f"segmentator_015_{vol}",
            f"segmentator_wider_015_{vol}",
        ]
        for meshname in meshes:
            dataname = f"{vol}_CG"
            print(meshname)
            path = os.path.join(results_folder, meshname, dataname, element_inits)
            discretizations = [
                pth for pth in os.listdir(path) if "dt0.01" in pth and "_pr2" in pth
            ]
            if len(discretizations) > 1:
                print(
                    "WARNING: multiple result folders, taking the first one aphabetically"
                )
                discretizations.sort()
            try:
                discretization = discretizations[0]
                path = os.path.join(path, discretization, reg_solver)
                T = None
                dt = None
                for text_piece in discretization.split("_"):
                    if "T" in text_piece:
                        T = float(text_piece.replace("T", ""))
                    if "dt" in text_piece:
                        dt = float(text_piece.replace("dt", ""))
                if T is None or dt is None:
                    raise ValueError("T and/or dt values not found in the path!")
                print(T, dt)
                nsteps = ceil(T / dt) + 1
                v_list, mesh = read_data_h5(
                    os.path.join(path, "ns_opt.h5"), meshname, "v", nsteps
                )
                Q = fd.FunctionSpace(mesh, "CG", 1)
                volume = fd.assemble(fd.project(fd.Constant(1.0), Q) * fd.dx)
                area_wall = fd.assemble(
                    fd.project(fd.Constant(1.0), Q) * fd.ds(marks["wall"])
                )
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
                        os.path.join(
                            "postprocessing", f"v_on_wall_{meshname}.{format}"
                        ),
                        bbox_inches="tight",
                    )
                fig.clf()

                fig, ax = plt.subplots()
                ax.plot(t_list, relative)
                ax.grid()
                ax.set_xlabel("time (s)", fontsize=15)
                ax.set_ylabel(
                    r"$\frac{||v||_{L^1(\Gamma_{wall})}|\Omega|}{||v||_{L^1(\Omega)}|\Gamma_{wall}|}$",
                    fontsize=15,
                )
                fig.savefig(os.path.join("postprocessing", f"relative_{meshname}.png"))
                fig.clf()
            except:
                print("results not found")
