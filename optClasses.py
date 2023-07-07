import pdb

import torch  # в коде на данный момент не реализованы вычисления на GPU
import numpy as np
import json
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from itertools import combinations
from torch.autograd import Variable
from itertools import product

from arguments import get_args
import pdb
import tqdm

pinf = 1e6

no_improve_steps = 100
min_lr = 1e-10
comp_heu = 1000
penalty_coef = 5.0

eps = 1e-6
eps2 = eps ** 2

messages_iter = 1000
almost_inf = 100.0

eps = 1e-5
sqrt2 = 2 ** 0.5

eps = 1e-5
sqrt2 = 2 ** 0.5


def remap_pts(reg, remap):
    for i in range(len(reg)):
        for j in range(len(reg[i])):
            reg[i][j] = remap[reg[i][j]]
    return reg[i][j]


def eq_pts_torus(p1, p2):
    diff = p1 - p2
    return np.all(np.abs(diff - np.round(diff)) < eps)


class OptDiagramTorus:  # создание диаграммы, вычисление минимизируемой функции
    def pt_index_torus(self, p, points):
        for i in range(points.shape[0]):
            if eq_pts_torus(p, points[i, :]):
                return i
        return -1

    def set_mask(self):
        self.mask = np.zeros((self.batch_size, self.n, self.n), dtype=bool)
        for _ in range(self.batch_size):
            for p in self.regions[_]:
                for i, j in combinations(p, 2):
                    self.mask[_, i, j] = True
        self.mask = torch.BoolTensor(self.mask).to(self.device)

    def __init__(
            self,
            points,
            batch_size,
            device,
    ):
        self.mask = None
        self.batch_size = batch_size
        self.device = device
        x = torch.tensor(points)
        xl = []
        for v in product([-1.0, 0.0, 1.0], repeat=2):
            v = torch.tensor(v)
            xl.append(x + v)
        X = torch.cat(xl, dim=-2)  # [9n, 2] --> [B, 9n, 2]
        vor = [Voronoi(X.detach().numpy()[_]) for _ in range(self.batch_size)]
        self.regions = [list() for i in range(self.batch_size)]
        for _ in range(self.batch_size):
            for i in range(len(vor[_].points)):
                if np.all((0 <= vor[_].points[i]) & (vor[_].points[i] < 1)):
                    self.regions[_].append(vor[_].regions[vor[_].point_region[i]])
        self.vertices = np.array([
            vor[_].vertices[np.all((0 <= vor[_].vertices) & (vor[_].vertices < 1), axis=1)]
            for _ in range(self.batch_size)
        ])  # [2n, 2] --> [B, 2n, 2]
        self.n = self.vertices.shape[1]
        remap = [{} for i in range(self.batch_size)]
        for _ in range(self.batch_size):
            for i in range(self.n):
                for j in range(len(vor[_].vertices)):
                    if eq_pts_torus(self.vertices[_][i], vor[_].vertices[j, :]):
                        remap[_][j] = i
        self.ok = True
        for _ in range(self.batch_size):
            for i in range(len(self.regions[_])):
                for j in range(len(self.regions[_][i])):
                    if self.regions[_][i][j] in remap[_].keys():
                        self.regions[_][i][j] = remap[_][self.regions[_][i][j]]
                    else:
                        self.ok = False
        self.set_mask()

    def forward(self, x, get_diams=False):
        xl = []
        for i in range(self.n):
            for v in torch.tensor(list(product([-1, 0, 1], repeat=2)), device=self.device):
                xl.append(x[..., i, :] + v)
        X = torch.stack(xl).permute(1, 0, 2).to(self.device)
        x2 = torch.square(X)
        x2s = torch.sum(x2, -1)
        distm = (
            -2 * X.matmul(X.transpose(-2, -1))
            + x2s.unsqueeze(-1)
            + x2s.unsqueeze(-2)
        )
        dist_torus = torch.min(
            distm.unfold(1, 9, 9).unfold(2, 9, 9).reshape((self.batch_size, self.n, self.n, 81)), dim=-1
        ).values
        dist_torus_x = dist_torus * self.mask
        squared_diams = torch.max(torch.max(dist_torus_x, dim=-1).values, dim=-1).values
        if get_diams:
            return torch.sum(squared_diams), squared_diams.detach().cpu().numpy() ** 0.5
        else:
            return torch.sum(squared_diams), None


class OptPartitionTorus:  # поиск оптимального разбиения, мультистарт
    def __init__(
        self,
        d,
        n,
        n_iter_circ,
        n_iter_part,
        lr_start,
        lr_decay,
        precision_opt,
        diam_tolerance,
        messages,
        may_plot,
        batch_size,
        device,
    ):
        self.best_diam = float('inf')
        self.best_points = None
        self.od = None
        self.n_iter1 = n_iter_circ  # максимальное число итераций при оптимизации упаковки кругов
        self.n_iter2 = n_iter_part  # максимальное число итераций при оптимизации разбиения
        self.n = n
        self.d = d
        self.lr_start = lr_start  # начальное значение learning rate
        self.lr_decay = lr_decay  # learning rate домножается на это число через каждые 1000 итераций
        self.messages = messages
        # 1 - только результаты запусков и лучший результат
        # 2 - результаты оптимизации через 1000 итераций
        # 3 - рисунки для результатов каждого запуска
        self.diam_tolerance = diam_tolerance  # отображаются диаметры, принадлежащие интервалу (diam_tolerance*d_max,d_max)
        self.precision_opt = precision_opt  # множитель learning rate для "точной" оптимизации
        self.best_p = None
        self.best_poly = None
        self.may_plot = may_plot
        self.batch_size = batch_size
        self.device = device
        self.history_packing = []
        self.history_partition = []
        self.precision_metrics = []

    def random_packing_torus(self):  # упаковка кругов в тор
        x = torch.rand((self.batch_size, self.n, self.d)).to(self.device)
        n1 = self.n * 3 ** self.d
        lr = 0.0003

        optimizer = torch.optim.Adam([x.requires_grad_()], lr=lr)
        mask = torch.tril(torch.ones(n1, n1, dtype=torch.bool), diagonal=-1).to(self.device)
        for i in tqdm.tqdm(range(1, self.n_iter1 + 1)):
            optimizer.zero_grad()
            xl = []
            for v in torch.tensor(list(product([-1, 0, 1], repeat=2)), device=self.device):
                xl.append(x + v)
            X = torch.cat(xl, dim=-2).to(self.device)  # [n1, 2], [b, n1, 2]
            # Attention above about batch_size

            # p[i] - p[j] = (x_i - x_j)^2 + (y_i - y_j)^2 =
            # = x_i^2 + x_j^2 - 2 x_i x_j + y_i^2 + y_j^2 - 2 y_i y_j
            x2s = torch.sum(X ** 2, -1)
            distm = (
                -2 * X.matmul(X.transpose(-2, -1))
                + x2s.unsqueeze(-1)
                + x2s.unsqueeze(-2)
            )
            distm = torch.where(mask, distm, float('inf'))
            y = torch.sum(-torch.min(torch.min(distm, dim=-1).values, dim=-1).values)
            self.history_packing.append(abs(y.cpu().item()) ** 0.5 / 2)
            y.backward()
            optimizer.step()
        if self.may_plot:
            plt.plot(self.history_packing)
            plt.title("Packing of Circles")
            plt.show()
        self.history_packing.clear()
        return x.cpu().detach().numpy()

    def random_partition_torus(self, x0):  # оптимизация разбиения
        # self.od = OptDiagramNd(self.poly, x0)
        self.od = OptDiagramTorus(x0, batch_size=self.batch_size, device=self.device)
        if not self.od.ok:
            return almost_inf, None
        x = torch.tensor(self.od.vertices).to(self.device)
        lr = self.lr_start
        x = x.requires_grad_()
        optimizer = torch.optim.Adam([x], lr=lr)
        y_best = almost_inf
        no_improve = 0
        precise = False
        for i in tqdm.tqdm(range(1, self.n_iter2 + 1)):
            #  100 == 1:
            #     self.od.vertices = x[0].detach().cpu().numpy()
            #     plot_partition(self, diam_tolerance=self.diam_tolerance)
            optimizer.zero_grad()
            y, _ = self.od.forward(x)
            y.backward()
            optimizer.step()
            ycur = y.cpu().detach().numpy()
            self.history_partition.append(ycur ** 0.5)
            if ycur < y_best:
                y_best = ycur
                no_improve = 0
            else:
                no_improve += 1
            if i % messages_iter == 0:
                if self.messages >= 1:
                    print(i, "   d = ", float(ycur) ** 0.5)
            if no_improve > no_improve_steps:
                lr = lr * self.lr_decay
                if lr < min_lr:
                    break
                if (ycur - self.best_diam) / lr > comp_heu:
                    break
                for g in optimizer.param_groups:
                    g["lr"] = lr
        loss, diams = self.od.forward(x.detach(), get_diams=True)
        self.od.vertices = x.cpu().detach().numpy()
        if self.may_plot:
            plt.plot(self.history_partition)
            plt.title("Partition of Torus")
            plt.show()
        self.history_partition.clear()
        return diams, self.od.vertices

    def multiple_runs(self, m):  # мультистарт, хранение лучшего разбиения
        for i in range(m):
            x = self.random_packing_torus()
            if np.isnan(x).any():
                if self.messages >= 1:
                    print("NaN in data")
                continue
            diams, points = self.random_partition_torus(x)

            new_rec = ""
            if self.od.regions is None:
                continue
            true_diams = np.array([
                find_true_diam(points[_], self.od.regions[_]) for _ in range(self.batch_size)
            ])
            index_min = np.argmin(true_diams)
            if points is not None and true_diams[index_min] < self.best_diam:
            # if points is not None and diams[index_min] < self.best_diam:
                self.best_diam = true_diams[index_min]
                self.best_points = points[index_min].copy()
                self.best_poly = self.od.regions[index_min].copy()
                new_rec = "***"
            if self.messages >= 1:
                print("run {0}/{1}, d_max = {2}  {3}".format(i + 1, m, self.best_diam, new_rec))
        self.od.vertices = self.best_points.copy()
        self.od.regions = self.best_poly.copy()
        return float(self.best_diam)

    def from_file(self, filename):  # для тора не работает
        with open(filename, "r") as f:
            l = f.readlines()
            poly = json.loads(l[3])
            points = json.loads(l[4])
            self.od = OptDiagramNd(self.poly, saved=(poly, points))
            self.od.set_mask()
            # d, dlist = self.od.diams()
            self.best_diam = d.detach().numpy()
            return self.best_diam

    def to_file(self, filename):  # запись в файл
        # TODO: переделать в json
        with open(filename, "w") as f:
            f.write(str(self.n) + "\n")
            f.write(str(self.best_diam) + "\n")  # максимальный диаметр
            f.write(str(len(self.od.vertices)) + "\n")  # число вершин разбиения
            json.dump(self.od.regions, f)  # полигоны, списки вершин
            f.write("\n")
            json.dump(self.od.vertices.tolist(), f)  # вершины

    def to_string(self):
        res = ""
        res += str(self.n) + "\n"
        res += str(self.best_diam) + "\n"
        res += str(len(self.od.vertices)) + "\n"  # число вершин разбиения
        res += json.dumps(self.od.regions)  # полигоны, списки вершин
        res += "\n"
        res += json.dumps(self.od.vertices.tolist())  # вершины
        return res

    def diams(self):  # для тора не работает
        best_diam = self.best_diam
        res = []
        for p in self.od.regions:
            for i, j in combinations(p, 2):
                curr_diam = np.linalg.norm(self.od.vertices[i] - self.od.vertices[j])
                if self.diam_tolerance * best_diam < curr_diam:
                    res.append((i, j))
        return res


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
        pcur = np.array(pcur[-len(p):])
        for i in range(len(pcur) - 1, -1, -1):
            mp = find_nearest(pcur[i], pcur[(i + 1) % len(pcur)])
            pcur[i] = mp.copy()
        pcur = pcur[-len(p):]
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
        find_true_diam=True,
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
        pcur = np.array(pcur[-part.n:])
        for i in range(len(pcur) - 1, -1, -1):
            mp = find_nearest(pcur[i], pcur[(i + 1) % len(pcur)])
            pcur[i] = mp.copy()
        pcur = pcur[-part.n:]
        # find honest diameter of part and relax with true_diam
        if find_true_diam:
            for i in range(len(pcur)):
                for j in range(i + 1, len(pcur)):
                    curr_diam = np.linalg.norm(pcur[i] - pcur[j])
                    true_diam = max(true_diam, curr_diam)
                    segments.append((pcur[i], pcur[j], curr_diam))
        if plot:
            midp = np.average(pcur, axis=0)
            plt.text(*midp, f"{len(pcur)}", fontsize=16)
            plt.fill(*zip(*pcur), alpha=0.4)
            # plt.fill(*zip(*[pts[i] for i in p]), alpha=0.4)
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
                print(d, true_diam, p, q)
                plt.plot([p[0], q[0]], [p[1], q[1]], alpha=0.5, linestyle="--")
                plt.text(
                    *((p + q) / 2),
                    f"{d / true_diam * 100:.8f}%",
                    fontsize=8,
                    alpha=0.5,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        plt.show()
    return true_diam
