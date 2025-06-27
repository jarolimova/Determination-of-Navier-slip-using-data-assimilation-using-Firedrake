from vmtk import vmtkscripts

from vtk import vtkImageData, vtkPolyData, vtkUnstructuredGrid

__all__ = [
    "marching_cubes",
    "surface_modeller",
    "surface_to_binary",
    "get_mesh",
    "mesh_to_surface",
    "surface_cell_to_point",
]


def marching_cubes(levelset: vtkImageData, level: float = 0.0) -> vtkPolyData:
    """Generate surface from level-th isosurface of the given image levelset

    Image -> Surface
    """
    marchingCubes = vmtkscripts.vmtkMarchingCubes()
    marchingCubes.Level = level
    marchingCubes.Image = levelset
    marchingCubes.Execute()
    return marchingCubes.Surface


def surface_modeller(surface: vtkPolyData, spacing: float = 1.0) -> vtkImageData:
    """Transform surface to distance image whose 0-th isosurface is the original surface

    Surface -> Image

    Args:
        spacing: voxel size of the created image
    """
    sm = vmtkscripts.vmtkSurfaceModeller()
    sm.SampleSpacing = spacing
    sm.Surface = surface
    sm.Execute()
    return sm.Image


def surface_to_binary(
    surface: vtkPolyData,
    spacing: float = 0.3,
    inside: float = 1.0,
    outside: float = 0.0,
) -> vtkImageData:
    """Create binary image marking inside and outside of the given surface

    Surface -> Image
    """
    stb = vmtkscripts.vmtkSurfaceToBinaryImage()
    stb.Surface = surface
    stb.InsideValue = inside
    stb.OutsideValue = outside
    stb.PolyDataToImageDataSpacing = [spacing, spacing, spacing]
    stb.Execute()
    return stb.Image


def get_mesh(
    surface: vtkPolyData,
    remesh_caps: bool = True,
    cap_edgelength: float = 1.0,
) -> vtkUnstructuredGrid:
    """Generate mesh from surface

    Surface -> Mesh
    """
    mesher = vmtkscripts.vmtkMeshGenerator()
    mesher.Surface = surface
    mesher.TargetEdgeLength = cap_edgelength
    if remesh_caps:
        mesher.RemeshCapsOnly = 1
    else:
        mesher.SkipRemeshing = 1
    mesher.Execute()
    return mesher.Mesh


def mesh_to_surface(mesh: vtkUnstructuredGrid) -> vtkPolyData:
    """Throws out volume elements from a mesh and return just the surface"""
    mts = vmtkscripts.vmtkMeshToSurface()
    mts.Mesh = mesh
    mts.Execute()
    return mts.Surface


def surface_cell_to_point(surface: vtkPolyData) -> vtkPolyData:
    """Project all cell data to point data"""
    ctp = vmtkscripts.vmtkSurfaceCellDataToPointData()
    ctp.Surface = surface
    ctp.Execute()
    return ctp.Surface
