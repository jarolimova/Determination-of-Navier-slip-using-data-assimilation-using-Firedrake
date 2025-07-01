""" Module for generating steady artificial MRI data from simulation results using Firedrake.
"""

import os
import json
import argparse
import numpy as np
from mpi4py import MPI
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import firedrake as fd
from math import ceil
from typing import List

from MRI_tools.MRI_firedrake import MRI
from data_loading import read_data_h5
from mri_volunteers import msh_to_h5


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataname", type=str, help="name of data")
    parser.add_argument("meshname", type=str, help="name of the mesh")
    parser.add_argument(
        "--element",
        type=str,
        default="p1p1",
        help="name of the element the data was computed with",
    )
    parser.add_argument(
        "--venc",
        type=float,
        default=None,
        help="VENC to be used for generation of noise - should be more than maximum velocity in any of the directions",
    )
    parser.add_argument(
        "--voxelsize",
        type=float,
        default=0.0025,
        help="voxelsize of the artificial MRI",
    )
    parser.add_argument(
        "--unsteady",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--msh_paths",
        type=str,
        nargs="*",
        help="paths to the msh files to be saved as h5",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="location of the data folder, default: data",
    )
    parser.add_argument(
        "--MRI_folder",
        type=str,
        default="MRI_npy",
        help="location of the MRI folder, default: MRI_npy",
    )
    args = parser.parse_args()
    return args


def make_noise(shape, venc=1.5, snr=5, seed=1111):
    """
    Generate noise with given shape and parameters based on the following formula:
    amount_of_noise = VENC / SNR
     standard_deviation = 0.45 * VENC / SNR

    source: https://doi.org/10.1017/jfm.2018.329

    Parameters:
        shape: The shape of the noise array.
        venc (float, optional): Velocity encoding. Defaults to 1.5.
        snr (float, optional): Signal-to-noise ratio. Defaults to 5.

    Returns:
        np.ndarray: Array containing generated noise.
    """
    # adding noise
    np.random.seed(seed)
    amount_of_noise = venc / snr
    standard_deviation = 0.45 * venc / snr
    noise = amount_of_noise * np.random.normal(size=shape, scale=standard_deviation)
    return noise


def setup_mri(mesh, voxelsize=0.0025, venc=None, timestep=0.0, padding=0):
    """
    Set up an MRI object based on the given mesh.

    Parameters:
        mesh: The mesh on which the MRI object will be based.
        voxelsize (float, optional): Voxel size for the MRI. Defaults to 0.0025.
        venc: velocity enconding parameter - max expected velocity magnitude (used for adding noise later)
        padding: number of voxel rows to add around as padding

    Returns:
        MRI: The MRI object.
    """
    mesh_coords = mesh.coordinates.dat.data_ro
    lcorner = np.array([mesh_coords[:, i].min() for i in range(3)]).astype(
        float
    ) - 2 * voxelsize * np.ones(3)
    rcorner = np.array([mesh_coords[:, i].max() for i in range(3)]).astype(
        float
    ) + 2 * voxelsize * np.ones(3)
    # find global bounding box
    glob_lcorner = [mesh.mpi_comm().allreduce(coord, MPI.MIN) for coord in lcorner]
    glob_rcorner = [mesh.mpi_comm().allreduce(coord, MPI.MAX) for coord in rcorner]
    # add padding
    glob_lcorner = [coord - padding * voxelsize for coord in glob_lcorner]
    glob_rcorner = [coord + padding * voxelsize for coord in glob_rcorner]
    # compute size of the grid
    size = np.array(
        [ceil((r - l) / voxelsize) for l, r in zip(glob_lcorner, glob_rcorner)],
        dtype=np.int16,
    )
    # create the mri object
    mri = MRI(
        size=size,
        origin=np.array(glob_lcorner),
        spacing=voxelsize * np.ones(3),
        timestep=timestep,
        venc=venc,
    )
    return mri


def data_to_mri(
    data_list: List[fd.Function],
    mri: MRI,
    snr=None,
) -> np.ndarray:
    """
    Generate artificial MRI data with optional noise.
    Noise is added if snr is not None.

    Parameters:
        data_vel (List[fd.Function]): Input data.
        mri (MRI): MRI object defining the grid to which the data should be loaded.
        snr (float, optional): Signal-to-noise ratio. Defaults to None.

    Returns:
        np.ndarray: Artificial MRI data projected to V.
    """
    # create mri function space
    mri_space = mri.create_mri_space(space_type="CG", hexahedral=False)
    # project simulated data to mri space
    mri_list = [fd.project(data_vel, mri_space) for data_vel in data_list]
    # extract numpy array
    np_array = mri.extract_array_from_mri_functions(mri_list)
    ideal_venc = abs(np_array).max()
    print(f"Ideal (minimal) venc: {ideal_venc}, chosen venc: {mri.venc}")
    if mri.venc < ideal_venc:
        print("venc is smaller than max velocity -> real case would have aliasing!")

    # add noise
    if snr is not None:
        noise = make_noise(np_array.shape, venc=mri.venc, snr=snr)
        np_array += noise
    return np_array


if __name__ == "__main__":
    args = get_args()

    # resave .msh meshes
    if args.msh_paths is not None:
        for pth in args.msh_paths:
            msh_to_h5(pth, folder=args.data_folder)

    # load h5 data from simulations on extended meshes
    data_path = os.path.join(
        args.data_folder, args.meshname, args.element, args.dataname
    )
    print(data_path + ".h5")
    if args.unsteady:
        with open(data_path + ".json", "r") as js:
            data_timelist = json.load(js)
            print(data_timelist)
        nsteps = len(data_timelist)
        data_list, mesh = read_data_h5(data_path + ".h5", args.meshname, nsteps=nsteps)
        pvdnsoptfile = fd.output.VTKFile(data_path + ".pvd")
        for i, dat in enumerate(data_list):
            dat.rename("v")
            pvdnsoptfile.write(dat, time=i)
    else:
        with fd.CheckpointFile(data_path + ".h5", "r") as h5file:
            mesh = h5file.load_mesh(args.meshname)
            data_list = [h5file.load_function(mesh, "v")]
        data_timelist = [0.0]

    steplen_list = [
        data_timelist[i + 1] - data_timelist[i] for i in range(len(data_timelist) - 1)
    ]
    avg_steplen = round(
        sum(steplen_list) / len(steplen_list) if len(steplen_list) > 0 else 0.0, 12
    )

    mri = setup_mri(
        mesh, voxelsize=args.voxelsize, venc=args.venc, timestep=avg_steplen, padding=2
    )
    for snr in [None, 3, 5]:
        mri.data = data_to_mri(
            data_list=data_list,
            mri=mri,
            snr=snr,
        )
        mri.results_folder = args.MRI_folder
        mri_name = args.dataname.split("_")[0] + "_" + args.meshname.split("_")[0]
        mri_name += f"_snr{snr}" if snr is not None else ""
        if args.voxelsize != 0.0025:
            mri_name += f"_{1000*float(args.voxelsize)}mm"
        mri.to_json(mri_name)
