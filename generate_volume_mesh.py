import os.path as pth
import json
import argparse
from gmsh_utils.gmsh_operations import generate_volume_mesh, convert_msh, mark_msh


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "json_file", type=str, help="Json file used to generate the surface mesh"
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="stl",
        help="Extension of file to be used for volume meshing, default: str",
    )
    parser.add_argument(
        "--hmin",
        type=float,
        default=None,
        help="Minimal edgelength for gmsh volume mesh generation",
    )
    parser.add_argument(
        "--hmax",
        type=float,
        default=None,
        help="Maximal edgelength for gmsh volume mesh generation",
    )
    args = parser.parse_args()
    return args


def generate_marked_msh(path, hmin=None, hmax=None):
    """generate volume in gmsh (if stl provided),
    mark msh volume and save it as xml as well
    """
    base, ext = pth.splitext(path)
    msh_path = base + ".msh"
    if ext == ".msh":
        # skip meshing because msh already has volume in it
        pass
    elif ext == ".stl":
        generate_volume_mesh(path, msh_path, hmin=hmin, hmax=hmax)
    else:
        raise NotImplementedError(f"Extension {ext} is not supported yet.")

    with open(base + "_cuts.json", "r") as js:
        bnd_data = json.load(js)
    mark_msh(msh_path, msh_path, bnd_data=bnd_data)
    convert_msh(msh_path, "xml")


if __name__ == "__main__":
    args = get_args()
    with open(args.json_file, "r") as jsnfile:
        parameters = json.load(jsnfile)
        name = parameters["name"]
        folder = parameters["folder"]
        scale = parameters["scale"]
        length = parameters["uniform_remeshing"]

    surface_path = pth.join(folder, name, f"{name}.{args.extension}")
    generate_marked_msh(surface_path, hmin=args.hmin, hmax=args.hmax)
