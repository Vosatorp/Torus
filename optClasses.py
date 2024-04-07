import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os

from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from itertools import combinations
from torch.autograd import Variable
from itertools import product

from utils import plot_partition, find_true_diam

from arguments import get_args
import pdb
import tqdm

no_improve_steps = 100
min_lr = 1e-10
comp_heu = 1000

messages_iter = 1000


def remap_pts(reg, remap):
    for i in range(len(reg)):
        for j in range(len(reg[i])):
            reg[i][j] = remap[reg[i][j]]
    return reg[i][j]


def eq_pts_torus(p1, p2):
    diff = p1 - p2
    return np.all(np.abs(diff - np.round(diff)) < 1e-6)


def pt_index_torus(p, points):
    for i in range(points.shape[0]):
        if eq_pts_torus(p, points[i, :]):
            return i
    return -1


class OptDiagramTorus:  # создание диаграммы, вычисление минимизируемой функции
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
            saved_data={}
    ):
        if saved_data:
            self.vertices = saved_data["vertices"]
            self.regions = saved_data["regions"]
            self.set_mask()
            return
        self.mask = None
        self.batch_size = batch_size
        self.device = device
        self.d = points.shape[-1]
        x = torch.tensor(points)
        xl = []
        for v in product([-1.0, 0.0, 1.0], repeat=self.d):
            v = torch.tensor(v)
            xl.append(x + v)
        X = torch.cat(xl, dim=-2)
        vor = [Voronoi(X.detach().numpy()[_]) for _ in range(self.batch_size)]
        self.regions = [list() for i in range(self.batch_size)]
        for _ in range(self.batch_size):
            for i in range(len(vor[_].points)):
                if np.all((0 <= vor[_].points[i]) & (vor[_].points[i] < 1)):
                    self.regions[_].append(vor[_].regions[vor[_].point_region[i]])
        self.vertices = np.array([
            vor[_].vertices[np.all((0 <= vor[_].vertices) & (vor[_].vertices < 1), axis=1)]
            for _ in range(self.batch_size)
        ])
        self.n = self.vertices.shape[1]
        # pdb.set_trace()
        remap = [{} for _ in range(self.batch_size)]
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
            for v in torch.tensor(list(product([-1, 0, 1], repeat=self.d)), device=self.device):
                xl.append(x[..., i, :] + v)
        X = torch.stack(xl).permute(1, 0, 2).to(self.device)
        x2 = torch.square(X)
        x2s = torch.sum(x2, -1)
        distm = (
                -2 * X.matmul(X.transpose(-2, -1))
                + x2s.unsqueeze(-1)
                + x2s.unsqueeze(-2)
        )
        k = 3 ** self.d
        dist_torus = torch.min(
            distm.unfold(1, k, k).unfold(2, k, k).reshape((self.batch_size, self.n, self.n, k * k)), dim=-1
        ).values
        dist_torus *= self.mask
        squared_diams = torch.max(torch.max(dist_torus, dim=-1).values, dim=-1).values
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
        self.best_diam = float("inf")
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
        pbar = tqdm.tqdm(total=self.n_iter1)
        for i in range(1, self.n_iter1 + 1):
            optimizer.zero_grad()
            xl = []
            for v in torch.tensor(list(product([-1, 0, 1], repeat=self.d)), device=self.device):
                xl.append(x + v)
            X = torch.cat(xl, dim=-2).to(self.device)

            # p[i] - p[j] = (x_i - x_j)^2 + (y_i - y_j)^2 =
            # = x_i^2 + x_j^2 - 2 x_i x_j + y_i^2 + y_j^2 - 2 y_i y_j
            x2s = torch.sum(X ** 2, -1)
            distm = (
                    -2 * X.matmul(X.transpose(-2, -1))
                    + x2s.unsqueeze(-1)
                    + x2s.unsqueeze(-2)
            )
            distm = torch.where(mask, distm, float("inf"))
            y = torch.sum(-torch.min(torch.min(distm, dim=-1).values, dim=-1).values)
            curr_avg_radius = (abs(y.cpu().item()) / self.batch_size) ** 0.5 / 2
            self.history_packing.append(curr_avg_radius)
            y.backward()
            optimizer.step()
            pbar.set_postfix(avg_radius=curr_avg_radius)
            pbar.update(1)
        if self.may_plot:
            plt.plot(self.history_packing)
            plt.title("Packing of Circles")
            plt.show()
        self.history_packing.clear()
        return x.cpu().detach().numpy()

    def random_partition_torus(self, x0):  # оптимизация разбиения
        self.od = OptDiagramTorus(x0, batch_size=self.batch_size, device=self.device)
        if not self.od.ok:
            return float("inf"), None
        x = torch.tensor(self.od.vertices).to(self.device)
        lr = self.lr_start
        x = x.requires_grad_()
        optimizer = torch.optim.Adam([x], lr=lr)
        y_best = float("inf")
        no_improve = 0
        pbar = tqdm.tqdm(total=self.n_iter2)
        for i in range(1, self.n_iter2 + 1):
            optimizer.zero_grad()
            y, _ = self.od.forward(x)
            y.backward()
            optimizer.step()
            y_curr = float(y.cpu().detach().numpy())
            curr_avg_max_diam = (y_curr / self.batch_size) ** 0.5
            self.history_partition.append(curr_avg_max_diam)
            pbar.set_postfix(avg_max_diam=curr_avg_max_diam)
            pbar.update(1)
            if y_curr < y_best:
                y_best = y_curr
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > no_improve_steps:
                lr = lr * self.lr_decay
                if lr < min_lr:
                    break
                if (y_curr - self.best_diam) / lr > comp_heu:
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
                self.best_diam = true_diams[index_min]
                self.best_points = points[index_min].copy()
                self.best_poly = self.od.regions[index_min].copy()
                new_rec = "***"
            if self.messages >= 1:
                print("run {0}/{1}, d_max = {2}  {3}".format(i + 1, m, self.best_diam, new_rec))
        self.od.vertices = self.best_points.copy()
        self.od.regions = self.best_poly.copy()
        return float(self.best_diam)

    def get_data(self):
        return {
            "n": self.n,
            "best_diam": self.best_diam,
            "len_vertices": len(self.od.vertices),
            "regions": self.od.regions,
            "vertices": self.od.vertices.tolist()
        }

    def save_to_file(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w") as f:
            json.dump(self.get_data(), f, indent=2)


class OptColoring:
    def __init__(
            self,

    ):
        self.dual_graph = None
        self.diagram = None
        self.colors = None

    def paint(self):
        """

        :param self:

        painting the diagram in accordance with the prohibition
        to be monochrome if the distance is no more than two in a dual graph

        :return:
        """
        self.colors = None
        pass

    def optimize_painting(self):
        # make random optimization
        pass

    def random_coloring(self, n):
        # packing n circles in rotated parallelepiped
        # build Voronoi diagram
        # paint dual_graph
        # optimize painting
        pass