import pdb
import os

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull


def find_nearest(point, target):
    best_point = point.copy()
    min_dist = 10.0
    corr_shift = np.array([0.0, 0.0])
    for shift in product([-1.0, 0.0, 1.0], repeat=point.shape[-1]):
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


def draw_cube(ax):

    # Координаты вершин куба
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ]

    # Список ребер куба
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ]

    # Рисование ребер куба
    for edge in edges:
        x = [vertices[edge[0]][0], vertices[edge[1]][0]]
        y = [vertices[edge[0]][1], vertices[edge[1]][1]]
        z = [vertices[edge[0]][2], vertices[edge[1]][2]]
        ax.plot(x, y, z, 'k')

    # Настройка осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def get_correct_partitions(pts, regions):
    if pts is None:
        return float("inf")
    corr_regions = []
    corr_regions_shifts = []
    dim = pts.shape[-1]
    print("dim:", dim)
    for p in regions:
        start_i = 0
        min_dist = 10.0
        center = np.array([0.5] * dim)
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
        corr_shifts = [np.array([0.0] * dim)]
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
    return max(np.max(pdist(polygon)) for polygon in polygons)


def plot_partition(
    part,
    diam_tolerance,
    filename=None,
):
    """
    Рисует разбиения тора так чтобы части выглядели нормально,
    находя нужные экземляры точек
    """
    dim = part.od.vertices.shape[-1]
    fig = plt.figure(figsize=(10, 10))
    if dim == 3:
        ax = plt.axes(projection='3d')
    polygons, shifts = get_correct_partitions(part.od.vertices, part.od.regions)
    true_diam = max(np.max(pdist(polygon)) for polygon in polygons)
    diameters = []
    for polygon in polygons:
        center = np.average(polygon, axis=0)
        if dim == 2:
            plt.text(*center, f"{len(polygon)}", fontsize=16)
            plt.fill(*zip(*polygon), alpha=0.4)
        else:
            color = [np.random.random(), np.random.random(), np.random.random()]
            hull = ConvexHull(polygon)
            ax.plot_trisurf(
                polygon[:, 0],
                polygon[:, 1],
                polygon[:, 2],
                triangles=hull.simplices,
                alpha=0.4,
                edgecolor=color
            )
        diameters.extend([
            (polygon[i], polygon[j])
            for i in range(len(polygon)) for j in range(i + 1, len(polygon))
            if np.linalg.norm(polygon[i] - polygon[j]) > true_diam * diam_tolerance
        ])
    if dim == 2:
        draw_square()
        plt.axis("equal")
    else:
        draw_cube(ax)
    print(f" n: {part.n}   diam: {part.best_diam:.8}   true_diam: {true_diam:.8f}")
    if dim == 2:
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
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename)


def diameters_to_julia(part):
    """

    :return:
        variables: (x_1, y_1), ..., (x_n, y_n)
        diameters_indexes: (i_1, j_1), ..., (i_m, j_m)
        shifts (+-1): (si_1, sj_1), ..., (si_m, sj_m)
    """
    polygons, shifts = get_correct_partitions(part.od.vertices, part.od.regions)
    true_diam = max(np.max(pdist(polygon)) for polygon in polygons)
    diameters_indexes = []
    shifts_result = []
    for polygon, shift_region in zip(polygons, shifts):
        for i in range(len(polygon)):
            for j in range(i + 1, len(polygon)):
                if np.linalg.norm(polygon[i] - polygon[j]) > true_diam * diam_tolerance:
                    diameters_indexes.append((i, j))
                    shifts_result.append((shift_region[i], shift_region[j]))
    return variables, diameters_indexes, shift_results
