import numpy as np
from math import log, ceil
import matplotlib.pyplot as plt

from vmtk_utils.vtk_wrapper import get_point_data 
from vmtk_utils.input_output import read_surface, write_surface, read_image
from vmtk_utils.surface_properties import surface_properties, surface_distance, surface_curvature
from vmtk_utils.surface_operations import surface_bool_operation, remesh_surface, smooth_surface, cap_surface, scale_surface, surface_projection, transform_to_ras, add_extensions

def smoothing_quality(original, smooth, volume=True, distance=True, curvature=True, overlap=False, decimals=2):
    print("QUALITY CHECK:")
    area_orig, vol_orig, _ = surface_properties(original)
    area_sm, vol_sm, _ = surface_properties(smooth)
    if area_sm == 0.0:
        "No surface! (zero area)"
        return None

    res = dict()
    if volume:
        print("Volume:")
        rel_volume = vol_sm / vol_orig
        res["original_volume"] = vol_orig
        res["smoothed_volume"] = vol_sm
        res["relative_volume"] = rel_volume
        print(f"    Original: {round(vol_orig, decimals)}")
        print(f"    Smoothed: {round(vol_sm, decimals)}")
        print(f"    Relative: {round(rel_volume*100, decimals)}%")

    if distance:
        print("Minimal surface distance:")
        surf = surface_distance(smooth, original)
        dist = get_point_data(surf, 'Distance')
        mean = np.mean(dist)
        median = np.median(dist)
        maximum = np.max(dist)
        minimum = np.min(dist)
        std = np.std(dist)
        res["mean_distance"] = mean
        res["median_distance"] = median
        res["maximal_distance"] = maximum
        res["minimal_distance"] = minimum
        # more decimals in case of small numbers:
        dec_mean = ceil(-log(mean,10)) if 0 < mean < 1. else 0
        dec_max = ceil(-log(maximum,10)) if 0 < maximum < 1. else 0
        print(f"    Mean:    {round(mean, decimals+dec_mean)}")
        print(f"    Maximal: {round(maximum, decimals+dec_max)}")

    if curvature:
        print("Mean absolute curvature:")
        try:
            surf_curv_orig = surface_curvature(original, absolute=True)
            surf_curv_sm = surface_curvature(smooth, absolute=True)
            array_orig = get_point_data(surf_curv_orig, "Curvature")
            array_sm = get_point_data(surf_curv_sm, "Curvature")
            curv_orig = np.mean(array_orig)
            curv_sm = np.mean(array_sm)
            rel_curvature = curv_sm/curv_orig
            res["original_curvature"] = curv_orig
            res["smoothed_curvature"] = curv_sm
            res["relative_curvature"] = rel_curvature
            dec_orig = ceil(-log(curv_orig,10)) if curv_orig < 1. else 0
            dec_sm = ceil(-log(curv_sm,10)) if curv_sm < 1. else 0
            print(f"   Original: {round(curv_orig, decimals+dec_orig)}")
            print(f"   Smoothed: {round(curv_sm, decimals+dec_sm)}")
            print(f"   Relative Drop: {round((1.-rel_curvature)*100, decimals)}%")
        except TypeError:
            res["relative_curvature"] = 0
            print("    Catched error in vmtk!")

    if overlap:
        print("WARNING: boolean operations are broken in VTK library!")
        intersection = surface_bool_operation(original, smooth, "intersection")
        _, vol_int, _ = surface_properties(intersection)
        overlap = vol_int / vol_orig
        res["overlap"] = overlap
        print(f"Overlap: {round(overlap*100, decimals)}%")
    print("")
    return res


def quality_plot(res_list, path='noname'):
    params = ["relative_volume", "relative_curvature"]
    x = np.arange(len(res_list))
    for p in params:
        y = np.array([q[p] for q in res_list])
        plt.plot(x, y, 'o')
        plt.savefig(f"{path}_{p}.png")
        plt.clf()
    # minimal distance plot:
    dist = ["minimal_distance", "maximal_distance", "median_distance", "mean_distance"]
    for p in dist:
        y = np.array([q[p] for q in res_list])
        plt.plot(x,y,'o', label=p)
    plt.savefig(f"{path}_distance.png")
    plt.clf()
    return


if __name__ == "__main__":

    surface = read_surface("working_files/quality_check/test_segmentace.stl")
    smooth1 = read_surface("working_files/quality_check/test_smooth1.stl")
    smooth2 = read_surface("working_files/quality_check/test_smooth2.stl")

    smoothing_quality(surface, smooth1)
    smoothing_quality(surface, smooth2)
