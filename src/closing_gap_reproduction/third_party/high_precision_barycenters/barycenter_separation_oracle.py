from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, polygonize_full, snap
import numpy as np
from scipy.spatial import cKDTree
from power_diagram_construction import get_power_diagram_truncated

import itertools
import time
import copy
import math
import random

# # CGAL wrapper to manipulate arrangements of lines.
# # Use my "hacked" version that allows for insertion of multiple segments at once.
# import skgeom as sg
# from skgeom import Segment2, Point2

# # Scipy library for KDtrees computation
from scipy.spatial import cKDTree

# # Plotting imports
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt

# from power_diagram_construction import get_power_diagram_truncated


def get_tuples_to_check_from_power_diagram_intersections(supports, wts):

    k = len(supports)

    # 1) Normalize weights exactly like your code
    wts = copy.deepcopy(wts)
    for i in range(k):
        wts[i] += -np.min(wts[i])
        wts[i] = wts[i] * k
        assert np.all(wts[i] >= 0)

    # 2) Bounding box
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    for i in range(k):
        S = supports[i]
        xmin = min(xmin, np.min(S[:, 0]))
        ymin = min(ymin, np.min(S[:, 1]))
        xmax = max(xmax, np.max(S[:, 0]))
        ymax = max(ymax, np.max(S[:, 1]))
    tol = 1.0
    xmin, ymin, xmax, ymax = xmin - tol, ymin - tol, xmax + tol, ymax + tol
    bbox_poly = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    # 3) Collect truncated power-diagram segments (unchanged logic)
    tot_seg_list = []
    Rs = []
    for i in range(k):
        S = supports[i]
        R = np.sqrt(np.asarray(wts[i]))   # radii = sqrt(weights)
        Rs.append(R)

        # Virtual sites to avoid infinite edges
        M = 4 * max(np.max(np.abs(S)), np.max(R))
        virtualS = np.asarray([[0, 4*M], [-4*M, 0], [0, -4*M], [4*M, 0]])
        virtualR = np.asarray([0, 0, 0, 0])
        S_aug = np.concatenate((S, virtualS))
        R_aug = np.concatenate((R, virtualR))

        segs = get_power_diagram_truncated(S_aug, R_aug, ((xmin, ymin), (xmax, ymax)))
        tot_seg_list.extend(segs)

    # 4) Shapely arrangement (fixed & more robust)
    lines = [LineString([seg[0], seg[1]]) for seg in tot_seg_list]

    # Add bounding box edges
    bx = list(bbox_poly.exterior.coords)
    box_edges = [LineString([bx[j], bx[(j+1) % (len(bx)-1)]]) for j in range(len(bx)-1)]

    # Node all intersections, light snap to avoid near-misses
    merged = unary_union(lines + box_edges)
    merged = snap(merged, bbox_poly, 1e-12)

    # polygonize_full returns (polygons, dangles, cuts, invalids)
    polys, _, _, _ = polygonize_full(merged)

    # Keep bounded faces strictly inside the bbox (and not equal to the bbox itself)
    faces = []
    for poly in polys.geoms if hasattr(polys, "geoms") else [polys]:
        inter = poly.intersection(bbox_poly)
        if not inter.is_empty and inter.area > 0 and not inter.equals_exact(bbox_poly, 1e-12):
            # explode multipolygons just in case
            if inter.geom_type == "MultiPolygon":
                faces.extend([p for p in inter.geoms if p.area > 0])
            elif inter.geom_type == "Polygon":
                faces.append(inter)

    # 5) One interior sample per face
    pts_to_check = []
    for poly in faces:
        c = poly.representative_point()  # guaranteed inside
        pts_to_check.append([c.x, c.y, 0.0])

    if not pts_to_check:
        return set()

    # 6) KD-trees in the correct lifted 3D space (FIXED z)
    kdtrees = []
    for i in range(k):
        curr_n = len(supports[i])
        maxR = float(np.max(Rs[i])) if len(Rs[i]) else 0.0
        kdtreepts = np.empty((curr_n, 3), dtype=float)
        kdtreepts[:, 0] = supports[i][:, 0]
        kdtreepts[:, 1] = supports[i][:, 1]
        # Correct lifting: z_j = sqrt(maxR^2 - R_j^2), clipped at 0 for safety
        lift = maxR**2 - (Rs[i]**2)
        lift = np.maximum(lift, 0.0)
        kdtreepts[:, 2] = np.sqrt(lift)
        kdtrees.append(cKDTree(kdtreepts))

    np_pts = np.asarray(pts_to_check)
    num_pts = np_pts.shape[0]
    curr_tups = np.zeros((num_pts, k), dtype=np.int32)
    for i in range(k):
        _, curr_tups[:, i] = kdtrees[i].query(np_pts, k=1, p=2)

    # 7) Self-consistency filter (same as your code)
    centroids = np.zeros((num_pts, 3))
    for i in range(k):
        centroids[:, 0] += supports[i][curr_tups[:, i], 0]
        centroids[:, 1] += supports[i][curr_tups[:, i], 1]
    centroids /= k  # z stays 0

    new_tups = np.zeros((num_pts, k), dtype=np.int32)
    for i in range(k):
        _, new_tups[:, i] = kdtrees[i].query(centroids, k=1, p=2)

    eq_rows = np.where((curr_tups == new_tups).all(axis=1))[0]
    tuples_to_check = set(map(tuple, new_tups[eq_rows].tolist()))
    return tuples_to_check


# import itertools
# import math
# import numpy as np
# import time
# import random
# import copy

# # CGAL wrapper to manipulate arrangements of lines.
# # Use my "hacked" version that allows for insertion of multiple segments at once.
# import skgeom as sg
# from skgeom import Segment2, Point2

# # Scipy library for KDtrees computation
# from scipy.spatial import cKDTree

# # Plotting imports
# from matplotlib.collections import LineCollection
# from matplotlib import pyplot as plt

# from power_diagram_construction import get_power_diagram_truncated


# def get_tuples_to_check_from_power_diagram_intersections(supports, wts):

#     k = len(supports)

#     # Copy the weights and make them nonnegative.
#     wts = copy.deepcopy(wts)
#     for i in range(k):
#         wts[i] += -np.min(wts[i])
#         wts[i] = wts[i] * k # Rescale weights for convenience.
#         for wt in wts[i]:
#             assert(wt >= 0)

#     # Compute the bounding box around the points in the support
#     xmin, ymin = math.inf, math.inf
#     xmax, ymax = -math.inf, -math.inf
#     for i in range(k):
#         S = supports[i]
#         xmin = min(xmin, np.min(S[:,0]))
#         ymin = min(ymin, np.min(S[:,1]))
#         xmax = max(xmax, np.max(S[:,0]))
#         ymax = max(ymax, np.max(S[:,1]))

#     tol = 1
#     xmin = xmin - tol
#     ymin = ymin - tol
#     xmax = xmax + tol
#     ymax = ymax + tol
#     bounding_box = ((xmin, ymin), (xmax, ymax))

#     # Compute the power diagrams.
#     tot_seg_list = []
#     Rs = []
#     for i in range(k):
#         S = supports[i]
#         W = wts[i]

#         # Distance is d^2 - R^2
#         # We want cost d^2 - W
#         # So W = R^2, W >= 0 always
#         R = np.sqrt(np.asarray(W))
#         Rs.append(R)

#         # Add virtual points to avoid infinite line segment corner cases.
#         # These are far enough that they do not change the power diagram diagram within the bounding box,
#         # and they also mean that all the infinite segments are irrelevant.
#         M = 4*max(np.max(np.abs(S)), np.max(R))
#         virtualS = np.asarray([[0,4*M], [-4*M, 0], [0, -4*M], [4*M, 0]])
#         virtualR = np.asarray([0, 0, 0, 0])
#         S = np.concatenate((S, virtualS))
#         R = np.concatenate((R, virtualR))

#         powerdiagram_segments = get_power_diagram_truncated(S, R, bounding_box)
#         tot_seg_list.extend(powerdiagram_segments)


#     # Compute the arrangement from intersecting all of the power diagrams.
#     arr = sg.arrangement.Arrangement()
#     outer_box = [
#         Segment2(Point2(xmin, ymin), Point2(xmin, ymax)), Segment2(Point2(xmin, ymax), Point2(xmax, ymax)),
#         Segment2(Point2(xmax, ymax), Point2(xmax, ymin)), Segment2(Point2(xmax, ymin), Point2(xmin, ymin))
#     ]
#     for s in outer_box:
#         arr.insert(s)

#     corr_format_seg_list = [Segment2(Point2(seg[0][0], seg[0][1]), Point2(seg[1][0], seg[1][1])) for seg in tot_seg_list]

#     for seg in corr_format_seg_list:
#         arr.insert(seg)


#     # For each face of the intersection, find a point inside it.
#     pts_to_check = []
#     face_list = [f for f in arr.faces]
#     unbounded_face = arr.unbounded_face()

#     tot_edg = 0
#     for f in face_list:
#         if f == unbounded_face:
#             continue
#         e_list = []
#         eit = f.outer_ccb
#         centroid = [0, 0, 0]
#         first_e = next(eit)
#         j = 0
#         while True:
#             e = next(eit)
#             s = e.source().point()
#             sx, sy = s.x(), s.y() # Sadly, this is a bottleneck because the PyBind wrapper is not very fast.
#             centroid[0] += sx
#             centroid[1] += sy
#             j += 1
#             if e == first_e:
#                 break
#         tot_edg += j
#         centroid[0] /= j
#         centroid[1] /= j
#         pts_to_check.append(centroid)


#     # Use KD trees to extract the tuples from the faces of the diagram.
#     # In principle, one could instead do this with a linear-time sweep through the arrangement data structure, but
#     # in practice that takes a lot of time because the Python-C++ PyBind wrapper is not so fast.

#     # Construct the KD trees
#     kdtrees = []
#     for i in range(k):
#         curr_n = len(supports[i])
#         maxR = np.max(Rs[i])
#         kdtreepts = np.ndarray((curr_n,3))
#         kdtreepts[:,0] = supports[i][:,0]
#         kdtreepts[:,1] = supports[i][:,1]
#         kdtreepts[:,2] = np.sqrt(maxR ** 2 - Rs[i][:] ** 2)

#         tree = cKDTree(kdtreepts)
#         kdtrees.append(tree)

#     # Extract the tuples from the KD trees
#     tuples_to_check_set = set()
#     np_pts_to_check = np.asarray(pts_to_check)

#     num_pts_to_check = np_pts_to_check.shape[0]
#     curr_tups_to_check = np.zeros((num_pts_to_check, k), dtype=np.int32)
#     for i in range(k):
#         _, curr_tups_to_check[:,i] = kdtrees[i].query(np_pts_to_check, k=1, p=2)

#     centroids = np.zeros((num_pts_to_check,3))
#     for i in range(k):
#         centroids[:,0] += supports[i][curr_tups_to_check[:,i],0]
#         centroids[:,1] += supports[i][curr_tups_to_check[:,i],1]
#     centroids = centroids / k

#     new_tups_to_check = np.zeros((num_pts_to_check, k), dtype=np.int32)
#     for i in range(k):
#         _, new_tups_to_check[:,i] = kdtrees[i].query(centroids, k=1, p=2)
#     eq_rows = np.where((curr_tups_to_check == new_tups_to_check).all(axis=1))
#     tuples_to_check_set.update([tuple(x) for x in new_tups_to_check[eq_rows].tolist()])

#     return tuples_to_check_set


def get_tuple_cost_barycenter(supports, tup):
    avgpoint = np.zeros(2)
    k = len(tup)
    for i in range(k):
        avgpoint += supports[i][tup[i],0:2]
    avgpoint = avgpoint / k

    cst = 0
    for i in range(k):
        cst += (avgpoint[0] - supports[i][tup[i],0])**2
        cst += (avgpoint[1] - supports[i][tup[i],1])**2
    cst = cst / k

    return cst
