import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from petsc4py import PETSc

print = PETSc.Sys.Print

volunteers_folder = "/usr/users/jarolimova/MRI_simulation/volunteer_jsons"


@dataclass
class Volunteer:
    identifier: str
    venc: float
    meshes: List[str]
    mri_files: List[str]
    magnitude_file: str
    transform_file: Optional[str]
    morphology_file: str
    scale: float
    dealiasing: List[Dict]

    def to_json(self):
        path = f"{volunteers_folder}/volunteer_{self.identifier}.json"
        with open(path, "w") as jsnfile:
            json.dump(self.__dict__, jsnfile, indent=4)
        return

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as jsnfile:
            parameters = json.load(jsnfile)
        print(f"Reading a volunteer from {path}")
        return cls(**parameters)

    def __eq__(self, other):
        equal = True
        for key in self.__dict__.keys():
            if self.__dict__[key] != other.__dict__[key]:
                equal = False
                print(f"difference in {key}: ", self.__dict__[key], other.__dict__[key])
        return equal

    def create_MRI(self, MRI_class, extension: Optional[str] = None):
        file_format = os.path.splitext(self.mri_files[0])[-1]
        if file_format == "":
            if extension is not None:
                if "." not in extension:
                    extension = "." + extension
                file_format = extension
            else:
                raise ValueError(
                    "No extension found, specify it using the extension argument."
                )
        if file_format == ".mhd":
            mri = MRI_class.from_mhd(
                list(self.mri_files), venc=self.venc, scale=self.scale
            )
        elif file_format == ".vti":
            mri = MRI_class.from_vti(
                list(self.mri_files), venc=self.venc, scale=self.scale
            )
        elif file_format == ".dcm":
            mri = MRI_class.from_dicom(
                list(self.mri_files), venc=self.venc, scale=self.scale
            )
        else:
            raise ValueError(f"unsupported file format {file_format} provided.")
        # apply transformation given by the .txt file (apply registration)
        mri.apply_transformation(self.transform_file)
        # apply dealiasing to the data in the MRI object
        for d in self.dealiasing:
            mri.data = mri.dealiasing(**d)
        return mri


def mesh_to_vol_id(meshname):
    volunteer_files = os.listdir(volunteers_folder)
    for fle in volunteer_files:
        vol = Volunteer.from_json(os.path.join(volunteers_folder, fle))
        basenames = [os.path.basename(mesh) for mesh in vol.meshes]
        basenames.sort()
        if meshname in basenames:
            vol_id = os.path.splitext(fle)[0].split("_")[1]
            return vol_id
    print(f"Volunteer id for meshname {meshname} not found!")
    return None
