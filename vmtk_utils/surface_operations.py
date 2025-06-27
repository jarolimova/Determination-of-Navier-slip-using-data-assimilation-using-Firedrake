from vmtk import vmtkscripts
from .centerlines import get_centerlines
from .surface_properties import get_boundary_ref_systems
from .vtk_wrapper import get_points

from vtk import vtkPolyData
from typing import List


__all__ = [
    "smooth_surface",
    "scale_surface",
    "remesh_surface",
    "clip_surface",
    "branch_clip",
    "add_extensions",
    "transform_to_ras",
    "cap_surface",
    "surface_connectivity",
    "surface_projection",
    "remove_surface_kite",
    "surface_bool_operation",
]


def smooth_surface(
    surface: vtkPolyData,
    iterations: int = 30,
    method: str = "taubin",
    passband: float = 1.0,
    relaxation: float = 0.01,
    skip_remeshing: bool = False,
) -> vtkPolyData:
    """Apply set number of iteration of surface smoothing

    Args:
        iterations: number of iterations
        method:
            taubin: Taubin smoothing based on parameter passband
            laplace: Laplace smoothing based on parameter relaxation
        passband: the smaller the smoother
            0.03  -> previous setting for aorta
            0.005 -> very smooth surface
    """
    smoothing = vmtkscripts.vmtkSurfaceSmoothing()
    smoothing.Surface = surface
    smoothing.NumberOfIterations = iterations
    smoothing.Method = method
    smoothing.PassBand = passband
    smoothing.RelaxationFactor = relaxation
    smoothing.SkipRemeshing = skip_remeshing
    smoothing.Execute()
    return smoothing.Surface


def scale_surface(surface: vtkPolyData, factor: float = 1.0) -> vtkPolyData:
    """Scale the surface based on the given factor

    Used mainly for conversion of spatial units

    Args:
        factor: conversion of units [mm] -> [m] done using factor value 0.001
    """
    sscale = vmtkscripts.vmtkSurfaceScaling()
    sscale.Surface = surface
    sscale.ScaleFactor = factor
    sscale.Execute()
    return sscale.Surface


def remesh_surface(
    surface: vtkPolyData,
    edgelength: float = 1.0,
    edgelength_array: str = "",
    factor: float = 1.0,
    minedge: float = 0.0,
    maxedge: float = 1e16,
    exclude_ids: List[int] = [],
    preserve_edges: bool = False,
) -> vtkPolyData:
    """Remesh surface with quality triangles of the given size

    Args:
        edgelength_array: used (when specified) to determine size of elements based on their position
            overrides uniform edgelength specified by edgelength parameter!
        factor: usually used for scaling of the sizearray
        preserve_edges: whether the edges should be remeshed or not - creates overlapped triangles if capped afterwards!!!
    """
    # possibility to add more parameters!
    remesh = vmtkscripts.vmtkSurfaceRemeshing()
    remesh.Surface = surface
    if edgelength_array == "":
        remesh.ElementSizeMode = "edgelength"
        remesh.TargetEdgeLength = edgelength
    else:
        remesh.ElementSizeMode = "edgelengtharray"
        remesh.TargetEdgeLengthArrayName = edgelength_array
    remesh.TargetEdgeLengthFactor = factor
    remesh.MaxEdgeLength = maxedge
    remesh.MinEdgeLength = minedge
    # remesh.InternalAngleTolerance = 0.2
    if exclude_ids != []:
        remesh.ExcludeEntityIds = exclude_ids
        remesh.CellEntityIdsArrayName = "CellEntityIds"
    if preserve_edges:
        remesh.PreserveBoundaryEdges = 1
    remesh.Execute()
    return remesh.Surface


def clip_surface(
    surface: vtkPolyData, name: str = "", value: float = 0.0
) -> vtkPolyData:
    """Clip surface using array values or interactively

    Args:
        name: name of the clip array
            "" -> interactive clipping
    """
    clipper = vmtkscripts.vmtkSurfaceClipper()
    clipper.Surface = surface
    if name != "":
        clipper.Interactive = 0
        clipper.ClipArrayName = name
        clipper.ClipValue = value
    clipper.Execute()
    return clipper.Surface


def branch_clip(
    surface: vtkPolyData,
    centerlines: vtkPolyData,
    group_ids: List = [],
    clip_value: float = 0.0,
) -> vtkPolyData:
    """Clip ends of branches based on marked centerlines

    Args:
        group_ids: which ids corresponds to the end to be clipped
        clip_value: the bigger, the less it clipps of
    """
    clipper = vmtkscripts.vmtkBranchClipper()
    clipper.Surface = surface
    clipper.Centerlines = centerlines
    clipper.ClipValue = clip_value
    clipper.GroupIds = group_ids
    clipper.InsideOut = 1
    clipper.Execute()
    return clipper.Surface


def add_extensions(
    surface: vtkPolyData, centerlines: vtkPolyData = None
) -> vtkPolyData:
    """Add flow extensions to a clipped surface based on centerlines"""
    flowext = vmtkscripts.vmtkFlowExtensions()
    flowext.Surface = surface
    if centerlines is None:
        refsystem = get_boundary_ref_systems(surface)
        points = get_points(refsystem)
        flowext.Centerlines = get_centerlines(
            surface, seedselector="pointlist", source=[points[0]], target=points[1:]
        )
    else:
        flowext.Centerlines = centerlines
    flowext.ExtensionMode = "boundarynormal"
    flowext.AdaptiveExtensionLength = 1
    flowext.ExtensionRatio = 2
    flowext.Interactive = 0
    flowext.Execute()
    return flowext.Surface


def transform_to_ras(surface: vtkPolyData, rmatrix: List) -> vtkPolyData:
    """Transform surface segmented using VMTK to RAS coordinates

    Segmented surface transformed to ras corresponds to the same coordinates as surfaces created using itksnap
    """
    transform = vmtkscripts.vmtkSurfaceTransformToRAS()
    transform.XyzToRasMatrixCoefficients = rmatrix
    transform.Surface = surface
    transform.Execute()
    return transform.Surface


def cap_surface(surface: vtkPolyData) -> vtkPolyData:
    """Add cap to the holes of the surface"""
    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Interactive = 0
    capper.Method = "simple"
    capper.Execute()
    return capper.Surface


def surface_connectivity(surface: vtkPolyData) -> vtkPolyData:
    """Extract largest connected part of the surface

    (There is a possiblility to implement other extraction methods)
    """
    conn = vmtkscripts.vmtkSurfaceConnectivity()
    conn.Surface = surface
    conn.Execute()
    return conn.Surface


def surface_projection(
    surface: vtkPolyData, reference_surface: vtkPolyData
) -> vtkPolyData:
    """Project all point data from reference surface onto the surface based on minimum distance criterion"""
    proj = vmtkscripts.vmtkSurfaceProjection()
    proj.Surface = surface
    proj.ReferenceSurface = reference_surface
    proj.Execute()
    return proj.Surface


def remove_surface_kite(surface: vtkPolyData, factor: float = 1.0) -> vtkPolyData:
    """Remove small kites in a surface to avoid Taubin smoothing artifacts

    criterium for adjustment of the point is:
        tol < local_average_area < factor * averate_area
    factor gives good results approximately in range (0.5, 2.5)
    """
    skr = vmtkscripts.vmtkSurfaceKiteRemoval()
    skr.Surface = surface
    skr.SizeFactor = factor
    skr.Execute()
    return skr.Surface


def surface_bool_operation(
    surface1: vtkPolyData, surface2: vtkPolyData, operation: str
) -> vtkPolyData:
    """Performs one of the following operations on the surfaces and returns the result

    Args:
        operation: "union","intersection","difference"
    WARNING: ERROR IN VTK - sometimes it doesnt work
    """
    boolean = vmtkscripts.vmtkSurfaceBooleanOperation()
    boolean.Surface = surface1
    boolean.Surface2 = surface2
    boolean.Operation = operation
    print(f"Creating {operation} of two surfaces ...")
    boolean.Execute()
    return boolean.Surface
