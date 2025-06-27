""" This script generates an artificial morphology and segmentation data from a given high-resolution mesh.
"""

import os
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import firedrake as fd
from data_loading import read_mesh_h5
from mri_artificial_data import setup_mri

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("meshname", type=str, help="name of the mesh to be used")
    parser.add_argument(
        "--voxelsize",
        type=float,
        default=0.001,
        help="size of the voxels in meters (e.g., 0.001 for 1 mm), used for an artificial MRI",
    )
    parser.add_argument(
        "--mesh_folder",
        type=str,
        default="data",
        help="folder where the mesh in .h5 format is located",
    )
    parser.add_argument(
        "--vtk_folder",
        type=str,
        default="data",
        help="folder where the mesh in .h5 format is located",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    threshold = 0.5

    mesh = read_mesh_h5(args.meshname, data_folder=args.mesh_folder)
    Q = fd.FunctionSpace(mesh, "CG", 1)
    identity = fd.project(fd.Constant(1.0), Q)

    mri = setup_mri(mesh, voxelsize=args.voxelsize, padding=2)
    # change mri to scalar functions
    mri.data = np.zeros((mri.number_of_timesteps, 1, *mri.size))
    mri_space = mri.create_mri_space(space_type="CG", hexahedral=False)
    artificial_image = fd.project(identity, mri_space)
    mri.data = mri.extract_array_from_mri_functions([artificial_image])

    # get data and make segmentations
    data_array = mri.data[0][0]
    segmentation_array = (data_array >= threshold).astype(float)

    # Convert NumPy array to vtkArray
    vtk_data_array = numpy_to_vtk(
        data_array.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )
    vtk_segmentation_array = numpy_to_vtk(
        segmentation_array.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )

    # Create vtkImageData object
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(data_array.shape)
    image_data.SetSpacing(*mri.spacing)  # Set the voxel spacing
    image_data.SetOrigin(*mri.origin)  # Set the origin

    segmentation_data = vtk.vtkImageData()
    segmentation_data.SetDimensions(segmentation_array.shape)
    segmentation_data.SetSpacing(*mri.spacing)
    segmentation_data.SetOrigin(*mri.origin)

    # Add the VTK array as the scalar field
    image_data.GetPointData().SetScalars(vtk_data_array)
    segmentation_data.GetPointData().SetScalars(vtk_segmentation_array)
    print("Dimensions:", image_data.GetDimensions())
    print("Number of Points:", image_data.GetNumberOfPoints())
    print("Number of Cells:", image_data.GetNumberOfCells())

    # save to vtk
    if not os.path.exists(args.vtk_folder):
        os.makedirs(args.vtk_folder, exist_ok=True)
    writer_vtk = vtk.vtkDataSetWriter()
    writer_vtk.SetFileName(
        os.path.join(args.vtk_folder, f"{args.meshname}_res{args.voxelsize}.vtk")
    )
    writer_vtk.SetInputData(image_data)
    writer_vtk.Write()

    writer_vtk.SetFileName(
        os.path.join(
            args.vtk_folder, f"{args.meshname}_res{args.voxelsize}_segmentation.vtk"
        )
    )
    writer_vtk.SetInputData(segmentation_data)
    writer_vtk.Write()
