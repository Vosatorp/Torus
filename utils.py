import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial.distance import pdist


def find_nearest(point, target):
    best_point = point.copy()
    min_dist = 10.0
    corr_shift = np.array([0.0, 0.0])
    for shift in product([-1.0, 0.0, 1.0], repeat=2):
        cur_p = point + np.array(shift)
        dist = np.linalg.norm(cur_p - target)
        if dist < min_dist:
            min_dist = dist
            best_point = cur_p.copy()
            corr_shift = shift
    return best_point, corr_shift


def draw_square():
    plt.plot([0, 1], [0, 0], "k")
    plt.plot([0, 0], [0, 1], "k")
    plt.plot([1, 1], [0, 1], "k")
    plt.plot([0, 1], [1, 1], "k")


def get_correct_partitions(pts, regions):
    if pts is None:
        return float("inf")
    corr_regions = []
    corr_regions_shifts = []
    for p in regions:
        start_i = 0
        min_dist = 10.0
        center = np.array([0.5, 0.5])
        for i in range(len(p)):
            if p is None:
                return float("inf")
            if p[i] is None:
                return float("inf")
            if pts[p[i]] is None:
                return float("inf")
            cur_p, _ = find_nearest(pts[p[i]], center)
            dist = np.linalg.norm(cur_p - center)
            if dist < min_dist:
                min_dist = dist
                start_i = i
        p_pre = pts[p[start_i]]
        corr_polygon = [p_pre]
        corr_shifts = [np.array([0.0, 0.0])]
        for i in p[start_i + 1:] + p[:start_i]:
            nearest_point, shift = find_nearest(pts[i], p_pre)
            corr_polygon.append(nearest_point)
            corr_shifts.append(shift)
            p_pre = nearest_point.copy()
        corr_polygon = np.array(corr_polygon[-len(p):])
        curr_shifts = np.array(corr_shifts[-len(p):])
        for i in range(len(corr_polygon) - 1, -1, -1):
            nearest_point, shift = find_nearest(
                corr_polygon[i],
                corr_polygon[(i + 1) % len(corr_polygon)]
            )
            corr_polygon[i] = nearest_point
            curr_shifts[i] = shift
        corr_polygon = corr_polygon[-len(p):]
        corr_regions.append(corr_polygon)
        corr_regions_shifts.append(curr_shifts)
    return corr_regions, corr_regions_shifts


def find_true_diam(pts, regions):
    if pts is None:
        return float("inf")
    polygons, shifts = get_correct_partitions(pts, regions)
    true_diam = 0
    for polygon in polygons:
        true_diam = max(true_diam, np.max(pdist(polygon)))
    return true_diam


def plot_partition(
    part,
    diam_tolerance,
    plot=True,
    filename=None,
):
    """
    Рисует разбиения тора так чтобы части выглядели нормально,
    находя нужные экземляры точек
    """
    if plot:
        fig = plt.figure(figsize=(10, 10))
    polygons, shifts = get_correct_partitions(part.od.vertices, part.od.regions)
    true_diam = find_true_diam(part.od.vertices, part.od.regions)
    diameters = []
    for polygon in polygons:
        if plot:
            midp = np.average(polygon, axis=0)
            plt.text(*midp, f"{len(polygon)}", fontsize=16)
            plt.fill(*zip(*polygon), alpha=0.4)
            diameters.extend([
                (polygon[i], polygon[j])
                for i in range(len(polygon)) for j in range(i + 1, len(polygon))
                if np.linalg.norm(polygon[i] - polygon[j]) > true_diam * diam_tolerance
            ])
    if plot:
        draw_square()
        plt.axis("equal")
        # plt.xlim(-0.2, 1.2)
        # plt.ylim(-0.2, 1.2)
        print(f" n: {part.n}   diam: {part.best_diam:.8}   true_diam: {true_diam:.8f}")
        plt.title(
            f" n: {part.n}   diam: {part.best_diam:.8f}   true_diam: {true_diam:.8f}",
            fontsize=16,
        )
        # draw diams
        for p, q in diameters:
            d = np.linalg.norm(p - q)
            plt.plot([p[0], q[0]], [p[1], q[1]], alpha=0.5, linestyle="--")
            text_coords = (p + q) / 2
            text_angle = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])) % 180
            plt.text(
                *text_coords,
                f"{d / true_diam * 100:.4f}%",
                fontsize=8,
                alpha=0.5,
                horizontalalignment="center",
                verticalalignment="center",
                rotation=text_angle,
            )
        plt.show()
        if filename is not None:
            plt.savefig(filename)
    return true_diam
