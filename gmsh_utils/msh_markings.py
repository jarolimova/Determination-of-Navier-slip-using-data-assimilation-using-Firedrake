import gmsh
from math import pi

__all__ = ["mark_msh"]


def mark_msh(surface_path, output_path, bnd_data, threshold_angle=pi / 3):
    """add markings to a generated volume mesh saved in msh format at {surface_path}.msh
    and save it to output file

        surface_path: path to the volume mesh (in msh format) without the extension .msh
        output_path: where to save the marked
        bnd_data: dictionary containing the boundary data from _cuts.json
        threshold_angle: threshold angle for splitting boundaries

    """
    gmsh.initialize()
    # read mesh to be marked
    gmsh.merge(surface_path)
    # split surfaces based on the thrashold angle - used to separate inlet and outlets
    gmsh.model.mesh.classifySurfaces(threshold_angle)
    # collect entities
    all_surfaces = gmsh.model.getEntities(dim=2)
    all_volumes = gmsh.model.getEntities(dim=3)
    # collect tags
    surface_tags = [surface[1] for surface in all_surfaces]
    volume_tags = [volume[1] for volume in all_volumes]
    print("surface tags: ", surface_tags, ", volume_tags: ", volume_tags)
    # mark volume with 0
    gmsh.model.addPhysicalGroup(dim=3, tags=volume_tags, tag=0)
    ends = [bnd_data["in"]] + bnd_data["outs"]
    # loop over ends and mark each one
    for i, end in enumerate(ends):
        # find tags of boundaries
        bndbox = create_bounding_box(end)
        box_ents = gmsh.model.getEntitiesInBoundingBox(*bndbox, dim=2)
        tags = [surface[1] for surface in box_ents]
        # create mark of given end (in: 2, outs: 3,4,...)
        gmsh.model.addPhysicalGroup(dim=2, tags=tags, tag=i + 2)
        # remove used tags
        for tag in tags:
            surface_tags.remove(tag)
    # tag all untagged parts with 1 (wall)
    gmsh.model.addPhysicalGroup(dim=2, tags=surface_tags, tag=1)
    gmsh.write(output_path)
    gmsh.finalize()


def create_bounding_box(end_data, factor=1.5):
    r_in = end_data["radius"]
    cp_in = end_data["point"]
    xmin, ymin, zmin = [cp - factor * r_in for cp in cp_in]
    xmax, ymax, zmax = [cp + factor * r_in for cp in cp_in]
    return (xmin, ymin, zmin, xmax, ymax, zmax)
