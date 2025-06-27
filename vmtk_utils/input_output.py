from vmtk import vmtkscripts

from vtk import vtkImageData, vtkPolyData, vtkUnstructuredGrid

__all__ = [
    "read_image",
    "read_surface",
    "read_mesh",
    "write_image",
    "write_surface",
    "write_mesh",
    "view_image",
    "view_surface",
]


def read_image(impath: str) -> vtkImageData:
    imreader = vmtkscripts.vmtkImageReader()
    imreader.InputFileName = impath
    imreader.Execute()
    return imreader.Image, imreader.XyzToRasMatrixCoefficients


def read_surface(spath: str) -> vtkPolyData:
    sreader = vmtkscripts.vmtkSurfaceReader()
    sreader.InputFileName = spath
    sreader.Execute()
    return sreader.Surface


def read_mesh(mpath: str) -> vtkUnstructuredGrid:
    mreader = vmtkscripts.vmtkMeshReader()
    mreader.InputFileName = mpath
    mreader.Execute()
    return mreader.Mesh


def write_image(image: vtkImageData, filename: str):
    iwriter = vmtkscripts.vmtkImageWriter()
    iwriter.Image = image
    iwriter.OutputFileName = filename
    iwriter.Execute()
    return


def write_surface(surface: vtkPolyData, filename: str):
    swriter = vmtkscripts.vmtkSurfaceWriter()
    swriter.Surface = surface
    swriter.OutputFileName = filename
    swriter.Execute()
    return


def write_mesh(mesh: vtkUnstructuredGrid, filename: str):
    mwriter = vmtkscripts.vmtkMeshWriter()
    mwriter.Mesh = mesh
    mwriter.OutputFileName = filename
    mwriter.Compressed = 0
    mwriter.Execute()
    return


def view_image(image: vtkImageData):
    iviewer = vmtkscripts.vmtkImageViewer()
    iviewer.Image = image
    iviewer.Execute()
    return


def view_surface(surface: vtkPolyData):
    sviewer = vmtkscripts.vmtkSurfaceViewer()
    sviewer.Surface = surface
    sviewer.Execute()
    return
