import os
import numpy as np
import json
from math import ceil
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from vmtk_utils.centerlines import (
    get_centerlines,
    mark_centerlines,
    resample_centerlines,
    smooth_centerlines,
)
from vmtk_utils.vtk_wrapper import get_points, get_point_data, add_scalar_function_array
from vmtk_utils.input_output import read_surface, write_surface
from vmtk_utils.conversion import surface_cell_to_point
from vmtk_utils.surface_operations import clip_surface, surface_connectivity

from typing import List, Dict, Tuple
from vtk import vtkPolyData

__all__ = ["centerline_clip"]


class CuttingEquation:
    """Function R^3 -> R whose zeroth levelset are local planes each define using point, normal and radius

    ni = normals, pi = points, ri = radii:
    f(x) = ni.(x-pi)  if |x-pi| <= ri for some i
         = 10       otherwise

    Args:
        points: list of points (numpy arrays) in R^3 defining the planes
        normals: list of normals (numpy arrays) in R^3 defining the planes
        radii: list of floats - each define the largest sphere (with center in point) inscribed to the artery
        radius_factor: float scales radii so that the sphere contains the cut plane
    """

    def __init__(
        self,
        points: List[np.ndarray],
        normals: List[np.ndarray],
        radii: List[float],
        radius_factor: float = 1.5,
    ):
        assert (
            len(points) == len(normals) == len(radii)
        ), "Points, normals and radii must have equal length"
        self.normals = normals
        self.points = points
        self.radii = [radius_factor * radius for radius in radii]

    def __call__(self, x: np.ndarray) -> float:
        value = 10.0
        x = np.array(x)
        for i in range(len(self.points)):
            distance = np.linalg.norm(x - self.points[i])
            if self.radii[i] is None or distance <= self.radii[i]:
                d = np.dot(self.points[i], self.normals[i])
                value = np.dot(self.normals[i], x) - d
        return value


def save_json(cuts, name, folder_path=""):
    with open(os.path.join(folder_path, f"{name}_cuts.json"), "w") as jsnfile:
        json.dump(cuts, jsnfile, indent=4)


def centerline_clip(
    surface: vtkPolyData,
    centerlines: vtkPolyData,
    inlet: float = 0.1,
    outlets: List[float] = [],
    radius_factor: float = 1.5,
    results_folder: str = "",
) -> Tuple[vtkPolyData, Dict]:
    """Locally cut surface using planes perpendicular to centerlines

    Args:
        inlet: what portion of centerline length much should be cut away at inlet
        outlets: what portion of centerline branch length should be cut at corresponding outlet
        radius_factor: scaling factor of MaximumInscribedSphereRadius to ensure that the spheres contain whole cross-section at every cutting location
    """
    # Resample to get smaller number of approximately equidistant points:
    centerlines = resample_centerlines(centerlines)
    # Smooth all the bumps on the centerline
    centerlines = smooth_centerlines(centerlines)
    # Create cell data CenterlineIds to mark branches:
    centerlines = mark_centerlines(centerlines)
    # Project CenterlineIds to points:
    centerlines = surface_cell_to_point(centerlines)

    # Get centerline coordinates and radii and divide them to groups based on CenterlineIds:
    centerline_ids = get_point_data(centerlines, "CenterlineIds")
    centerline_ids = centerline_ids.astype(np.int64, copy=False)
    num_ids = int(max(centerline_ids) + 1)
    centerline_coords = get_points(centerlines)
    centerline_coord_groups = [
        centerline_coords[centerline_ids == i] for i in range(num_ids)
    ]
    radius_data = get_point_data(centerlines, "MaximumInscribedSphereRadius")
    radius_groups = [radius_data[centerline_ids == i] for i in range(num_ids)]

    # Create splines corresponding to each group (Centerline Id)
    # Splines: parametric curve for centerline, its derivative, spline for radius values
    param_groups = [
        np.linspace(0.0, 1.0, num=len(group)) for group in centerline_coord_groups
    ]
    coord_splines = [
        CubicSpline(param_groups[i], centerline_coord_groups[i], extrapolate=False)
        for i in range(num_ids)
    ]
    derivatives = [spline.derivative() for spline in coord_splines]
    radius_splines = [
        CubicSpline(param_groups[i], radius_groups[i], extrapolate=False)
        for i in range(num_ids)
    ]

    # Plot splines:
    if results_folder != "":
        plot_3d_spline(coord_splines, results_folder)
        plot_spline(radius_splines, results_folder, name="radii")

    # First add values corresponding to inlet:
    print("Centerline Clip:")
    cuts = dict()
    cut_points, cut_normals, cut_radii = [], [], []
    point = coord_splines[0](inlet)
    normal = derivatives[0](inlet) / np.linalg.norm(derivatives[0](inlet))
    radius = radius_splines[0](inlet)
    print(f"IN: point: {point}, normal: {normal}, radius: {radius}")
    cuts["in"] = {
        "point": point.tolist(),
        "normal": normal.tolist(),
        "radius": radius.tolist(),
    }
    cut_points.append(point)
    cut_normals.append(normal)
    cut_radii.append(radius)

    # Add values for outlets:
    cuts["outs"] = []
    for i, outlet in enumerate(outlets):
        if outlet is not None:
            t = 1 - outlet
            point = coord_splines[i](t)
            # Outlet normals have to be flipped!
            normal = -derivatives[i](t) / np.linalg.norm(derivatives[i](t))
            radius = radius_splines[i](t)
            print(f"OUT{i+1}: point: {point}, normal: {normal}, radius: {radius}")
            cuts["outs"].append(
                {
                    "point": point.tolist(),
                    "normal": normal.tolist(),
                    "radius": radius.tolist(),
                }
            )
            cut_points.append(point)
            cut_normals.append(normal)
            cut_radii.append(radius)

    # Create cutting function, add it to surface, clip it and remove excess:
    cutting_function = CuttingEquation(
        cut_points, cut_normals, cut_radii, radius_factor
    )
    add_scalar_function_array(surface, cutting_function, "cutting_function")
    clip = clip_surface(surface, name="cutting_function")
    clip_clean = surface_connectivity(clip)

    # Save results if the folder is specified
    if results_folder != "":
        write_surface(centerlines, f"{results_folder}/centerlines.vtp")
        write_surface(surface, f"{results_folder}/cutting_function.vtp")
        write_surface(clip, f"{results_folder}/clipped.vtp")
    return clip_clean, cuts


def plot_spline(splines, folder_path, steps=100, name="data_plot"):
    time = np.linspace(0.0, 1.0, num=steps)
    for spline in splines:
        data = spline(time)
        plt.plot(time, data, "o-")
    plt.savefig(f"{folder_path}/{name}.png")
    plt.clf()
    return


def plot_3d_spline(splines, folder_path, steps=100, name="splines"):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    time = np.linspace(0.0, 1.0, num=steps)
    for spline in splines:
        coords = spline(time)
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])
    plt.savefig(f"{folder_path}/{name}.png")
    plt.clf()
    return


if __name__ == "__main__":
    name, points = ("stl_5_surface01", {"inlet": 0.1, "outlets": [0.05]})
    # name, points = ("test_segmentace", {"inlet":0.1, "outlets":[0.05]})
    # name, points = ("surface_ras_smooth01", {"inlet":0.1, "outlets":[0.08, 0.02]})

    print(f"name: {name}")
    cl_path = f"working_files/centerline_clip/centerlines_{name}.vtp"
    surf = read_surface(f"working_files/centerline_clip/{name}.vtp")

    cl = read_surface(cl_path)
    clip = centerline_clip(surf, cl, **points)

    write_surface(surf, f"working_files/centerline_clip/modified_{name}.vtp")
    write_surface(clip, f"working_files/centerline_clip/clipped_{name}.vtp")
