import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def find_true_diam(pts, regions):
    true_diam = 0
    if pts is None:
        return float("inf")
    for p in regions:
        pcur = []
        start_i = 0
        min_dist = 10.0
        target = np.array([0.5, 0.5])
        for i in range(len(p)):
            if p is None:
                return float("inf")
            if p[i] is None:
                return float("inf")
            if pts[p[i]] is None:
                return float("inf")
            cur_p = find_nearest(pts[p[i]])
            dist = np.linalg.norm(cur_p - target)
            if dist < min_dist:
                min_dist = dist
                start_i = i
        p_pre = pts[p[start_i]]
        pcur.append(p_pre)
        for i in p[start_i + 1:] + p[:start_i]:
            mp = find_nearest(pts[i], p_pre)
            pcur.append(mp.copy())
            p_pre = mp.copy()
        pcur = np.array(pcur[-len(p) :])
        for i in range(len(pcur) - 1, -1, -1):
            mp = find_nearest(pcur[i], pcur[(i + 1) % len(pcur)])
            pcur[i] = mp.copy()
        pcur = pcur[-len(p) :]
        # find honest diameter of part and relax with true_diam
        if find_true_diam:
            for i in range(len(pcur)):
                for j in range(i + 1, len(pcur)):
                    true_diam = max(true_diam, np.linalg.norm(pcur[i] - pcur[j]))
    return true_diam


def draw_square():
    plt.plot([0, 1], [0, 0], "k")
    plt.plot([0, 0], [0, 1], "k")
    plt.plot([1, 1], [0, 1], "k")
    plt.plot([0, 1], [1, 1], "k")


def find_nearest(point, target=np.array([0.5, 0.5])):
    best_points = point.copy()
    min_dist = 10.0
    for v in product([-1.0, 0.0, 1.0], repeat=2):
        cur_p = point + np.array(v)
        dist = np.linalg.norm(cur_p - target)
        if dist < min_dist:
            min_dist = dist
            best_points = cur_p.copy()
    return best_points


def plot_partition(
    part,
    diam_tolerance,
    is_find_true_diam=True,
    plot=True,
):
    """
    Рисует разбиения тора так чтобы части выглядели нормально,
    находя нужные экземляры точек
    """
    if plot:
        fig = plt.figure(figsize=(10, 10))
    pts = part.od.vertices
    true_diam = 0
    segments = []
    for p in part.od.regions:
        pcur = []
        start_i = 0
        min_dist = 10.0
        target = np.array([0.5, 0.5])
        for i in range(len(p)):
            cur_p = find_nearest(pts[p[i]])
            dist = np.linalg.norm(cur_p - target)
            if dist < min_dist:
                min_dist = dist
                start_i = i
        p_pre = pts[p[start_i]]
        pcur.append(p_pre)
        for i in p[start_i + 1:] + p[:start_i]:
            mp = find_nearest(pts[i], p_pre)
            pcur.append(mp.copy())
            p_pre = mp.copy()
        pcur = np.array(pcur[-part.n :])
        for i in range(len(pcur) - 1, -1, -1):
            mp = find_nearest(pcur[i], pcur[(i + 1) % len(pcur)])
            pcur[i] = mp.copy()
        pcur = pcur[-part.n :]
        # find honest diameter of part and relax with true_diam
        if is_find_true_diam:
            for i in range(len(pcur)):
                for j in range(i + 1, len(pcur)):
                    curr_diam = np.linalg.norm(pcur[i] - pcur[j])
                    true_diam = max(true_diam, curr_diam)
                    segments.append((pcur[i], pcur[j], curr_diam))
        if plot:
            midp = np.average(pcur, axis=0)
            plt.text(*midp, f"{len(pcur)}", fontsize=16)
            plt.fill(*zip(*pcur), alpha=0.4)
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
        for p, q, d in segments:
            if d > true_diam * diam_tolerance:
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
    return true_diam
