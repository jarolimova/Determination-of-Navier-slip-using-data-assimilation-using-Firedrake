import numpy as np

from vtk.util.numpy_support import numpy_to_vtk

from typing import Callable
from vtk import vtkPointSet, vtkDataSet

__all__ = ["get_points", "get_point_data", "add_scalar_function_array"]


def get_points(obj: vtkPointSet) -> np.ndarray:
    """get numpy array of point coordinates from vtkPointSet object

    vtkPointSet = vtkPolyData/vtkUnstructuredGrid/...
    """
    return np.array(obj.GetPoints().GetData())


def get_point_data(obj: vtkDataSet, name: str) -> np.ndarray:
    """get numpy array of point data values saved under the given name from vtkDataSet object

    vtkDataSet - vtkPolyData/vtkUstructuredGrid/...
    """
    return np.array(obj.GetPointData().GetArray(name))


def add_scalar_function_array(
    obj: vtkDataSet, function: Callable, name: str
) -> vtkDataSet:
    """add scalar function array to a vtkDataSet object

    vtkDataSet - vtkPolyData/vtkUstructuredGrid/...

    Args:
        function: scalar function which can evaluate on space coordinates (array of 3 values)
        name: name of the PointDataArray (as displayed in Paraview)
    """
    points = get_points(obj)
    arr = np.array([function(point) for point in points])
    array = numpy_to_vtk(arr, deep=True)
    array.SetName(name)
    dataset = obj.GetPointData()
    dataset.AddArray(array)
    dataset.Modified()
    return obj
