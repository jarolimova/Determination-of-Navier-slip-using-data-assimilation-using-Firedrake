from vmtk import vmtkscripts
from .input_output import write_surface
import numpy as np

from typing import Tuple
from vtk import vtkPolyData

__all__ = [
    "surface_properties",
    "surface_curvature",
    "surface_distance",
    "get_boundary_ref_systems",
    "compute_h_mean_on_surface",
    "compute_h_max_on_surface",
    "compute_h_min_on_surface",
]


def surface_properties(surface: vtkPolyData, printing: bool = False) -> Tuple:
    """Compute area, volume and shape index of a surface

    Args:
        printing: whether to print surface properties or not
    """
    massprop = vmtkscripts.vmtkSurfaceMassProperties()
    massprop.Surface = surface
    massprop.Execute()
    if printing:
        print(f"Surface Area: {round(massprop.SurfaceArea, 2)}")
        print(f"Volume: {round(massprop.Volume, 2)}")
        print(f"Shape Index: {round(massprop.ShapeIndex, 2)}")
    return (massprop.SurfaceArea, massprop.Volume, massprop.ShapeIndex)


def surface_curvature(
    surface: vtkPolyData,
    curvature_type: str = "mean",
    absolute: bool = False,
    reciprocal: bool = False,
) -> vtkPolyData:
    """Computes curvature of the surface and saves it as PointDataArray

    Args:
        curvature_type: "mean","gaussian","maximum","minimum"
    """
    curv = vmtkscripts.vmtkSurfaceCurvature()
    curv.CurvatureType = curvature_type
    curv.Surface = surface
    curv.AbsoluteCurvature = int(absolute)
    curv.BoundedReciprocal = int(reciprocal)
    curv.Execute()
    return curv.Surface


def surface_distance(
    surface: vtkPolyData, reference_surface: vtkPolyData
) -> vtkPolyData:
    """Computes minimal distance of surface from reference_surface and saves it to the surface as PointDataArray"""
    dist = vmtkscripts.vmtkSurfaceDistance()
    dist.Surface = surface
    dist.ReferenceSurface = reference_surface
    dist.DistanceArrayName = "Distance"
    dist.DistanceVectorsArrayName = "DistanceVectors"
    dist.SignedDistanceArrayName = "SignedDistance"
    dist.Execute()
    return dist.Surface


def get_boundary_ref_systems(surface: vtkPolyData, filename: str = "") -> vtkPolyData:
    bndref = vmtkscripts.vmtkBoundaryReferenceSystems()
    bndref.Surface = surface
    bndref.Execute()
    if filename != "":
        write_surface(bndref.ReferenceSystems, filename)
    return bndref.ReferenceSystems


def compute_h_mean_on_surface(surface: vtkPolyData):
    """
    Computes an average edge length on a surface mesh.
    """
    # h_array = np.array([])
    h_sum = 0
    k = 0
    num_cells = surface.GetNumberOfCells()
    for i in range(num_cells):
        triangle_cell = surface.GetCell(i)
        for j in range(3):
            k += 1
            b = triangle_cell.GetEdge(j).GetBounds()
            dist = np.sqrt((b[0] - b[1]) ** 2 + (b[2] - b[3]) ** 2 + (b[4] - b[5]) ** 2)
            h_sum += dist
            # h_array = np.append(dist, h_array)
    # h_mean = np.mean(h_array)
    h_mean = h_sum / k
    return h_mean


def compute_h_max_on_surface(surface: vtkPolyData):
    """
    Computes the maximum edge length in a surface mesh.
    """
    h_max = 0
    count = 0
    num_cells = surface.GetNumberOfCells()
    for i in range(num_cells):
        triangle_cell = surface.GetCell(i)
        num_edges = triangle_cell.GetNumberOfEdges()
        for j in range(num_edges):
            edge = triangle_cell.GetEdge(j)
            b = edge.GetBounds()
            dist = np.sqrt((b[0] - b[1]) ** 2 + (b[2] - b[3]) ** 2 + (b[4] - b[5]) ** 2)
            if dist > h_max:
                h_max = dist
                # print(h_max)
    return h_max


def compute_h_min_on_surface(surface: vtkPolyData):
    """
    Computes the minimum edge length in a surface mesh.
    """
    h_min = 1e5
    num_cells = surface.GetNumberOfCells()
    for i in range(num_cells):
        triangle_cell = surface.GetCell(i)
        num_edges = triangle_cell.GetNumberOfEdges()
        for j in range(num_edges):
            edge = triangle_cell.GetEdge(j)
            b = edge.GetBounds()
            dist = np.sqrt((b[0] - b[1]) ** 2 + (b[2] - b[3]) ** 2 + (b[4] - b[5]) ** 2)
            if dist < h_min:
                h_min = dist
                # print(h_min)
    return h_min
