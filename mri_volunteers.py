""" Module for processing MRI volunteer data (into JSON + .npy format) and converting mesh files to h5 format.
"""

import os
import shutil
import argparse
import firedrake as fd
from MRI_tools.MRI_firedrake import MRI
from MRI_tools.volunteer import Volunteer


def msh_to_h5(meshpath, folder="data"):
    # read mesh
    meshname = os.path.basename(meshpath)
    print(f"resaving mesh {meshname} as .h5 ...")
    mesh = fd.Mesh(meshpath + ".msh", name=meshname)
    # save the mesh as h5
    with fd.CheckpointFile(os.path.join(folder, f"{meshname}.h5"), "w") as h5file:
        h5file.save_mesh(mesh)
    # copy json for boundary markings
    if not os.path.exists(os.path.join(folder, f"{meshname}_cuts.json")):
        shutil.copy(meshpath + "_cuts.json", folder)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "json_paths",
        type=str,
        nargs="+",
        help="paths to the volunteer json files",
    )
    parser.add_argument(
        "--MRI_folder",
        type=str,
        default="MRI_npy",
        help="location of the MRI folder, default: MRI_npy",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    for json_path in args.json_paths:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file {json_path} does not exist.")
        vol = Volunteer.from_json(json_path)
        if vol.transform_file is not None:
            # create MRI object based on the Volunteer obj
            mri = vol.create_MRI(MRI)
            mri.results_folder = args.MRI_folder
            mri.to_json(f"vol{vol.identifier}_reg")
            vol.transform_file = None

        # create MRI object based on the Volunteer obj
        assert vol.transform_file is None
        mri = vol.create_MRI(MRI)
        mri.results_folder = args.MRI_folder
        mri.to_json(f"vol{vol.identifier}")
        for meshpath in vol.meshes:
            msh_to_h5(meshpath)
