"""This module implements functionality helpful for reading meshes and data (in .h5 format), loading MRI data (from JSON+npy files)
into MRI objects, and analyzing meshes. It also provides a way to save the MRI data to VTK files for inspection.
"""

import os
import copy
import argparse
import numpy as np
import firedrake as fd
from typing import Optional, List, Tuple
from MRI_tools.MRI_firedrake import MRI


def read_mesh_h5(meshname, data_folder="data"):
    """read h5 mesh saved using one of the mri_ data prep scripts to data_folder

    Args:
        meshname (str): name of the mesh to read, without the .h5 extension
        data_folder (str): folder where the mesh is stored, default: "data"
    """
    with fd.CheckpointFile(os.path.join(data_folder, meshname + ".h5"), "r") as h5:
        mesh = h5.load_mesh(meshname)
    return mesh


def read_data_h5(h5_path, meshname, dataname="data", nsteps=1, printing=True):
    """read data from h5 file, meshname is the name of the mesh in the h5 file,
    dataname is the name of the data to read, nsteps is the number of time steps to read.

    Args:
        h5_path (str): path to the h5 file
        meshname (str): name of the mesh in the h5 file
        dataname (str): name of the data to read, default: "data"
        nsteps (int): number of time steps to read, default: 1
        printing (bool): whether to print the progress, default: True
    """
    data_list = []
    if printing:
        print("reading h5 ...")
        print("h5_path: ", h5_path)
    with fd.CheckpointFile(h5_path, "r") as h5file:
        mesh = h5file.load_mesh(meshname)
        if printing:
            print(f"mesh {meshname} loaded ...")
        for i in range(nsteps):
            data = h5file.load_function(mesh, dataname, idx=i)
            data_list.append(data.copy(deepcopy=True))
            if printing:
                print(f"datapoint {dataname}_{i} loaded ...")
    return data_list, mesh


def load_to_mri(
    mri: MRI,
    mesh: Optional[fd.Mesh],
    padding: int = 2,
    timesteps: Optional[List[int]] = None,
    space_type="DG",
    hexahedral=True,
) -> Tuple[List[fd.Function], float]:
    """
    Reads MRI data from a JSON file, applies optional volunteer-specific dealiasing,
    and maps the MRI data to a finite element mesh.

    Parameters:
        mri (str):  the MRI object.
        mesh (fd.Mesh): The finite element mesh determining the region of interest (ROI).
        padding (int, optional): Padding size for the region of interest (ROI)
                                 computation. Defaults to 0.
        timesteps (Optional[List[int]], optional): List of timesteps to process.
                                                   If None, all timesteps are used.
        space_type (str, optional): Type of finite element space for mapping
                                     (e.g., "DG"). Defaults to "DG".
        hexahedral (bool, optional): Whether to use a hexahedral mesh structure.
                                     Defaults to True, False corresponds to
                                     tetrahedral structure.

    Returns:
        List[fd.Function]: A list of finite element functions representing the MRI
                           data mapped to the MRI mesh.

    Raises:
        FileNotFoundError: If the specified MRI or volunteer JSON files are not found.
        ValueError: If the data or configurations are invalid for processing.
    """

    if mesh is None:
        lcorner, rcorner = None, None
    else:
        lcorner, rcorner = mri.compute_roi(mesh, padding=padding)
    mri_space = mri.create_mri_space(
        lcorner, rcorner, space_type=space_type, hexahedral=hexahedral
    )

    mri_functions = mri.data_to_mesh(function_space=mri_space, timesteps=timesteps)
    return mri_functions


def analyze_mesh(mesh):
    """
    Compute volume of the mesh and area of the inlet

    Parameters:
        mesh (fd.Mesh): The finite element mesh to which MRI data will be mapped.

    Returns:
        volume and inlet area of the mesh
    """
    v = fd.FunctionSpace(mesh, "CG", 1)
    one = fd.project(fd.Constant(1.0), v)
    volume = fd.assemble(one * fd.dx)
    gamma_in = fd.assemble(one * fd.ds(2))
    num_vertices = v.dim()
    return volume, gamma_in, num_vertices


def generate_mask(
    mri: MRI,
    mesh: Optional[fd.Mesh],
    padding: int = 2,
    space_type="DG",
    hexahedral=True,
):
    """Generate a mask for the MRI data based on the mesh and padding.

    Args:
        mri (MRI): The MRI object containing the data.
        mesh (fd.Mesh, optional): The finite element mesh to determine the region of interest (ROI).
                                  If None, the ROI will be computed from the MRI data.
        padding (int, optional): Padding size for the ROI computation. Defaults to 2.
        space_type (str, optional): Type of finite element space for mapping (e.g., "DG").
                                    Defaults to "DG".
        hexahedral (bool, optional): Whether to use a hexahedral mesh structure. Defaults to True.
    """
    mri_copy = copy.deepcopy(mri)
    mri_copy.data = np.zeros((mri_copy.number_of_timesteps, 1, *mri_copy.size))
    lcorner, rcorner = mri_copy.compute_roi(mesh, padding=padding)
    mri_space = mri_copy.create_mri_space(
        lcorner, rcorner, space_type=space_type, hexahedral=hexahedral
    )
    mask = mri_copy.make_mask(mesh, mri_space, padding=2)
    mri.data = mri.apply_mask(mask)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "MRI_json", type=str, help="name of the MRI json file in MRI_json_folder"
    )
    parser.add_argument(
        "meshname", type=str, help="name of the mesh to load the data into"
    )
    parser.add_argument(
        "--element",
        type=str,
        default="p1p1",
        help="name of the element to be used when loading the data to the mesh",
    )
    parser.add_argument(
        "--MRI_json_folder",
        type=str,
        default="MRI_npy",
        help="location of the MRI_json_folder, default: MRI_npy",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="location of the data folder, default: data",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """read mesh and MRI data, analyze the mesh, and save the MRI data to VTK files for inspection."""
    args = get_args()
    mesh = read_mesh_h5(args.meshname, args.data_folder)
    volume, gamma_in, num_vertices = analyze_mesh(mesh)
    print(
        f"mesh: {args.meshname}, volume: {volume}, inlet surface: {gamma_in}, #vertices: {num_vertices}, #dofs: {4*num_vertices}"
    )
    mri = MRI.from_json(os.path.join(args.MRI_json_folder, args.MRI_json))
    dg_functions = load_to_mri(
        mri=mri,
        mesh=mesh,
        padding=1,
        space_type="DG",
        hexahedral=True,
    )
    pvd = fd.output.VTKFile(f"{args.MRI_json}_MRInpy.pvd")
    for i, fun in enumerate(dg_functions):
        pvd.write(fun, time=i)
