""" Module for processing MRI volunteer data (into JSON + .npy format) and converting mesh files to h5 format.
"""

import os
import shutil
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


if __name__ == "__main__":
    vol_ids = [
        "01",
        "03",
        "04",
        "10",
        "12",
        "13",
        "14",
        "17",
        "18",
        "19",
        "20", 
        "57",
        "59",
    ]
    for vol_id in vol_ids:
        vol = Volunteer.from_json(
            f"/usr/users/jarolimova/MRI_simulation/volunteer_jsons/volunteer_{vol_id}.json"
        )
        if vol.transform_file is not None:
            # create MRI object based on the Volunteer obj
            mri = vol.create_MRI(MRI)
            mri.results_folder = "MRI_npy"
            mri.to_json(f"vol{vol.identifier}_reg")
            vol.transform_file = None

        # create MRI object based on the Volunteer obj
        assert vol.transform_file is None
        mri = vol.create_MRI(MRI)
        mri.results_folder = "MRI_npy"
        mri.to_json(f"vol{vol.identifier}")
        for meshpath in vol.meshes:
            msh_to_h5(meshpath)
