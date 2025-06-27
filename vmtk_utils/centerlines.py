import numpy as np

from vmtk import vmtkscripts
from .input_output import write_surface

from vtk import vtkPolyData
from typing import List

__all__ = [
    "get_centerlines",
    "resample_centerlines",
    "smooth_centerlines",
    "mark_centerlines",
    "project_centerlines",
    "centerlines_network",
]


def get_centerlines(
    surface: vtkPolyData,
    seedselector: str = "pickpoint",
    source: List = [],
    target: List = [],
    filename: str = "",
) -> vtkPolyData:
    """Computes centerlines of a given surface

    Args:
        seedselector: method for selection of the seed
            pickpoint    -> interactively select
            openprofiles -> open profiles in the surface
            pointlist    -> list of sourcepoints and targetpoints
        source: list of source points for pointlist seedselector
        target: list of target points for pointlist seedselector
    """
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = surface
    centerlines.SeedSelectorName = seedselector
    if seedselector == "pointlist":
        centerlines.SourcePoints = list(np.ndarray.flatten(np.array(source)))
        centerlines.TargetPoints = list(np.ndarray.flatten(np.array(target)))
        print(f"Source Points: {centerlines.SourcePoints}")
        print(f"Target Points: {centerlines.TargetPoints}")
    centerlines.Execute()
    if filename != "":
        write_surface(centerlines.Centerlines, filename)
    return centerlines.Centerlines


def resample_centerlines(centerlines: vtkPolyData, filename: str = "") -> vtkPolyData:
    """resample centerlines so that the points are approximately equidistant"""
    resample = vmtkscripts.vmtkCenterlineResampling()
    resample.Centerlines = centerlines
    resample.Execute()
    if filename != "":
        write_surface(resample.Centerlines, filename)
    return resample.Centerlines


def smooth_centerlines(
    centerlines: vtkPolyData,
    iterations: int = 100,
    factor: float = 0.1,
    filename: str = "",
) -> vtkPolyData:
    """smooth centerlines with a moving average filter with given factor and number of iterations"""
    smoother = vmtkscripts.vmtkCenterlineSmoothing()
    smoother.Centerlines = centerlines
    smoother.NumberOfSmoothingIterations = iterations
    smoother.SmoothingFactor = factor
    smoother.Execute()
    if filename != "":
        write_surface(smoother.Centerlines, filename)
    return smoother.Centerlines


def mark_centerlines(centerlines: vtkPolyData, filename: str = "") -> vtkPolyData:
    """mark centerline ends (GroupIds) and split branches (CenterlineIds)"""
    marked_centerlines = vmtkscripts.vmtkEndpointExtractor()
    marked_centerlines.Centerlines = centerlines
    marked_centerlines.Execute()
    if filename != "":
        write_surface(marked_centerlines.Centerlines, filename)
    return marked_centerlines.Centerlines


def project_centerlines(surface: vtkPolyData, centerlines: vtkPolyData) -> vtkPolyData:
    """project centerline data onto surface points"""
    projection = vmtkscripts.vmtkSurfaceCenterlineProjection()
    projection.Surface = surface
    projection.Centerlines = centerlines
    projection.UseRadiusInformation = True
    projection.Execute()
    return projection.Surface


def centerlines_network(surface: vtkPolyData) -> vtkPolyData:
    """generate centerlines without specification of source and target

    might influence direction of flow of the centerlines
    """
    network = vmtkscripts.vmtkCenterlinesNetwork()
    network.Surface = surface
    network.Execute()
    return network.Centerlines
