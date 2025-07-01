import os
import sys
import json

import numpy as np
from typing import Union, Dict
from vtk import vtkPolyData, vtkUnstructuredGrid

from vmtk_utils.input_output import read_surface, write_surface, read_image
from vmtk_utils.surface_operations import (
    remesh_surface,
    smooth_surface,
    cap_surface,
    scale_surface,
    surface_projection,
    transform_to_ras,
    add_extensions,
)
from vmtk_utils.conversion import marching_cubes
from vmtk_utils.surface_properties import surface_curvature, get_boundary_ref_systems
from vmtk_utils.centerlines import get_centerlines, project_centerlines
from vmtk_utils.centerline_clip import centerline_clip, get_points, save_json
from vmtk_utils.quality_check import smoothing_quality


class Pipeline:
    def __init__(
        self,
        name: str = "noname",
        input_path: str = "noname",
        folder: str = "working_files/meshprep",
        folder_path: str = "working_files/meshprep/noname",
        uniform_remeshing: float = 0.0,
        centerlines_parameters: Union[Dict, str] = {"seedselector": "pickpoint"},
        initial_remeshing_parameters: Dict = None,
        smoothing_parameters: Dict = None,
        clip_parameters: Dict = None,
        extensions: bool = False,
        final_remeshing_parameters: Dict = None,
        cap_edgelength: float = 1.0,
        scale: float = 1.0,
    ):
        attributes = {key: value for key, value in locals().items() if key != "self"}
        print(attributes)
        for name, value in attributes.items():
            setattr(self, name, value)
        self.folder_path = f"{self.folder}/{self.name}"

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as jsnfile:
            parameters = json.load(jsnfile)
        print(f"Initializing a pipeline from {path}")
        return cls(**parameters)

    def to_json(self, path="."):
        with open(f"{path}/{self.name}.json", "w") as jsnfile:
            json.dump(self.__dict__, jsnfile, indent=4)
        print(f"Pipeline setting was saved to {path}/{self.name}.json")

    def __call__(self, surface: vtkPolyData = None) -> vtkUnstructuredGrid:
        # prepare folder for results
        if not os.path.isdir(self.folder_path):
            os.makedirs(self.folder_path)

        # save pipeline setting to the folder
        self.to_json(path=f"{self.folder_path}")

        # if surface not provided, read from input_path
        if surface is None:
            surface = self.read_surface()
        original = surface
        write_surface(surface, f"{self.folder_path}/original.vtp")

        if self.uniform_remeshing > 0.0:
            print("uniform remeshing ...")
            surface = remesh_surface(surface, edgelength=self.uniform_remeshing)
            write_surface(surface, f"{self.folder_path}/uniform.vtp")

        print("computing curvature ...")
        try:
            curvature = surface_curvature(surface, absolute=True, reciprocal=True)
        except:
            print("Curvate could not be computed!")

        print(self.centerlines_parameters, type(self.centerlines_parameters))
        if type(self.centerlines_parameters) is dict:
            print("computing centerlines ...")
            centerlines = get_centerlines(surface, **self.centerlines_parameters)
        elif type(self.centerlines_parameters) is str:
            print(f"reading centerlines from {self.centerlines_parameters}...")
            centerlines = read_surface(self.centerlines_parameters)
        else:
            raise ValueError(
                f"Invalid centerlines_parameters type: {self.centerlines_parameters}"
            )
        write_surface(centerlines, f"{self.folder_path}/centerlines.vtp")

        if self.initial_remeshing_parameters is not None:
            surface = self.remeshing(
                surface, self.initial_remeshing_parameters, curvature, centerlines
            )
            write_surface(surface, f"{self.folder_path}/initial_remesh.vtp")

        if self.smoothing_parameters is not None:
            print("smoothing ...")
            surface = smooth_surface(surface, **self.smoothing_parameters)
            write_surface(surface, f"{self.folder_path}/smooth.vtp")
        smoothing_quality(original, surface)

        if self.clip_parameters is not None:
            print("centerline cliping ...")
            surface, cuts = centerline_clip(
                surface, centerlines, **self.clip_parameters
            )
            write_surface(surface, f"{self.folder_path}/clipped.vtp")
            if self.extensions:
                print("adding extensions ...")
                surface = add_extensions(surface, centerlines=centerlines)
                write_surface(surface, f"{self.folder_path}/extensions.vtp")
                bnd_ref = get_boundary_ref_systems(surface)
                points = get_points(bnd_ref)
                for cut in [cuts["in"]] + cuts["outs"]:
                    point = cut["point"]
                    diffs = [p - point for p in points]
                    norms = [np.linalg.norm(diff) for diff in diffs]
                    cut["point"] = points[np.argmin(norms)].tolist()

            print("generating cuts ...")
            if self.scale != 1.0:
                for cut in [cuts["in"]] + cuts["outs"]:
                    cut["point"] = [self.scale * point for point in cut["point"]]
                    cut["radius"] = self.scale * cut["radius"]
            save_json(cuts, self.name, self.folder_path)

            print("generating refsystems ...")
            surface_refsys = scale_surface(surface, factor=self.scale)
            refsys = get_boundary_ref_systems(
                surface_refsys, f"{self.folder_path}/{self.name}_refsys.dat"
            )

        if self.final_remeshing_parameters is not None:
            surface = self.remeshing(
                surface, self.final_remeshing_parameters, curvature, centerlines
            )
            write_surface(surface, f"{self.folder_path}/final_remesh.vtp")

        print("capping surface ...")
        csurface = cap_surface(surface)
        surface = remesh_surface(
            csurface, exclude_ids=[1], edgelength=self.cap_edgelength
        )
        write_surface(surface, f"{self.folder_path}/capped_surface.vtp")

        print("scaling surface...")
        if self.scale != 1.0:
            surface = scale_surface(surface, factor=self.scale)
            write_surface(surface, f"{self.folder_path}/scaled_surface.vtp")

        self.save_surface(surface)
        return

    def read_surface(self):
        print("reading surface ...")
        _, extension = os.path.splitext(self.input_path)
        if extension in {".vtp", ".stl"}:
            surface = read_surface(self.input_path)
        elif extension == ".mha":
            segmentation, matrix = read_image(self.input_path)
            surface = transform_to_ras(marching_cubes(segmentation, level=0.5), matrix)
        elif extension == ".vtk":
            segmentation, matrix = read_image(self.input_path)
            surface = marching_cubes(segmentation, level=0.5)
        else:
            raise ValueError(f"Invalid file type: {extension}")
        return surface

    def remeshing(self, surface, parameters, curvature, centerlines):
        print("remeshing ...")
        if "edgelength_array" in parameters.keys():
            if parameters["edgelength_array"] == "Curvature":
                surface = surface_projection(surface, curvature)
                write_surface(surface, f"{self.folder_path}/curvature.vtp")
            elif parameters["edgelength_array"] == "MaximumInscribedSphereRadius":
                surface = project_centerlines(surface, centerlines)
                write_surface(surface, f"{self.folder_path}/radius.vtp")
        surface = remesh_surface(surface, **parameters)
        return surface

    def save_surface(self, surface):
        write_surface(surface, f"{self.folder_path}/{self.name}.stl")
        print(f"Number of surface triangles:  {surface.GetNumberOfCells()}")


if __name__ == "__main__":
    jsn = "mesh_preparation.json"
    args = sys.argv
    if len(args) > 1:
        jsn = args[1]

    pipeline = Pipeline.from_json(jsn)
    pipeline()
