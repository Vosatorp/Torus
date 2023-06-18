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

no_impr_steps = 100
min_lr = 1e-8
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


class OptDiagramTorus:  # создание диаграммы, вычисление минимизируемой функции
    def pt_index_torus(self, p, points):
        for i in range(points.shape[0]):
            if self.eq_pts_torus(p, points[i, :]):
                return i
        return -1

    def eq_pts_torus(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return abs(dx - round(dx)) < eps and abs(dy - round(dy)) < eps

    def remap_pts(self, reg, remap):
        for i in range(len(reg)):
            for j in range(len(reg[i])):
                reg[i][j] = remap[reg[i][j]]
        return reg[i][j]

    def set_mask(self):
        n = len(self.points)
        self.mask = np.zeros((n, n), dtype=bool)
        # self.mask = np.zeros((n * 9, n * 9), dtype=bool)
        # l9 = list(range(9))
        for p in self.polygons:
            for i, j in combinations(p, 2):
                self.mask[i, j] = True
                # for k, l in combinations(l9, 2):
                #    self.mask[i*9+k,j*9+l] = True
        self.mask = torch.BoolTensor(self.mask).to(self.device)

    def __init__(self, points=None, saved=None, penalty_coef=5.0, device="cpu"):
        self.penalty_coef = penalty_coef
        self.device = device
        if saved is None:
            x = torch.Tensor(points)
            xl = []
            for v in product([-1.0, 0.0, 1.0], repeat=2):
                v = torch.tensor(v)
                xl.append(x + v)
            X = torch.cat(xl)
            ext_pts = X.detach().numpy()
            vor = Voronoi(ext_pts)
            self.vor = vor
            regions = []
            for i in range(ext_pts.shape[0]):
                if 0 <= vor.points[i, 0] < 1 and 0 <= vor.points[i, 1] < 1:
                    regions.append(vor.regions[vor.point_region[i]])
            # print(regions)
            vertices = []
            for v in vor.vertices:
                if 0 <= v[0] < 1 and 0 <= v[1] < 1:
                    vertices.append(v)
            remap = {}
            for i in range(len(vertices)):
                for j in range(len(vor.vertices)):
                    if self.eq_pts_torus(vertices[i], vor.vertices[j, :]):
                        remap[j] = i
                        continue
                    # print("not in the sq :", vor.vertices[j,:])
            ok = True
            for i in range(len(regions)):
                for j in range(len(regions[i])):
                    if regions[i][j] in remap.keys():
                        regions[i][j] = remap[regions[i][j]]
                    else:
                        # print("*** no index: ",vor.vertices[regions[i][j],:])
                        ok = False
            # self.remap_pts(regions, remap)
        self.points = np.array(vertices)
        n = len(self.points)
        self.n = n
        self.polygons = regions
        self.ok = ok

        self.set_mask()
    #    self.mask_lines = np.zeros((m,n - self.bound_vert_n), dtype=bool)
    #    for i in range(dist_lines.shape[0]):
    #      for j in range(self.bound_vert_n, dist_lines.shape[1]):
    #       if abs(dist_lines[i,j])<eps:
    #          self.line_link.append([j,i])
    #          self.mask_lines[i,j - self.bound_vert_n] = True

    #    self.mask_lines =  torch.BoolTensor(self.mask_lines)
    # self.bound_pt_tensor = torch.Tensor(bound.points)
    # self.line_U = torch.tensor(bound.U)
    # self.line_v = torch.tensor(bound.v.reshape((len(bound.planes),1)))

    def forward(self, x):
        # x_all = torch.cat((self.bound_pt_tensor,x))
        # n = len(x_all)
        xl = []
        for i in range(x.shape[0]):
            for v in torch.tensor(list(product([-1, 0, 1], repeat=2)), device=self.device):
                xl.append(x[i, :] + v)
        X = torch.cat(xl).reshape(
            9 * self.n, 2
        ).to(self.device)  # 9 экзмепляров каждой точки со сдвигами
        # print(X)
        x2 = torch.square(X)
        x2s = torch.sum(x2, 1)
        distm = (
                -2 * X.mm(X.t())
                + x2s
                + x2s.reshape((self.n * 9, 1))
                + torch.eye(9 * self.n).to(self.device) * sqrt2
        )
        # print(distm.shape)
        # print(self.mask.shape)
        dist_torus = torch.min(
            distm.unfold(0, 9, 9).unfold(1, 9, 9).reshape((self.n, self.n, 81)), 2
        ).values
        # print(dist_torus)
        # diags = torch.sqrt(torch.masked_select(dist_torus,self.mask))
        diags = torch.masked_select(dist_torus, self.mask)
        # dist_lines = torch.abs(torch.masked_select(self.line_U.mm(x.t()) + self.line_v, self.mask_lines))
        # не самый экономный способ, вычисляется много лишних расстояний
        # self.err = torch.max(dist_lines)
        return torch.max(diags)  # + penalty_coef * self.err
        # второе слагаемое - "штрафная" функция; можно обойтись без нее, если задавать точки на ребрах по формуле p = a + \alpha(b-a)

class OptPartitionTorus:  # поиск оптимального разбиения, мультистарт
    def __init__(
        self,
        d,
        n,
        n_iter_circ,
        n_iter_part,
        lr_start=0.005,
        lr_decay=0.93,
        prec_opt=1 / 1000,
        diam_tol=0.99,
        messages=1,
        device="cpu",
    ):
        self.n_iter1 = n_iter_circ  # максимальное число итераций при оптимизации упаковки кругов
        self.n_iter2 = n_iter_part  # максимальное число итераций при оптимизации разбиения
        self.n = n
        self.d = d
        # self.d = len(points[0])             # число частей
        # self.poly = CPolytop(self.d, center = np.zeros(self.d), pts = points)
        # self.poly.prepare()
        # self.U = torch.Tensor(self.poly.U)
        # self.v = torch.Tensor(self.poly.v).reshape((len(self.poly.planes),1))
        self.lr_start = lr_start  # начальное значение learning rate
        self.lr_decay = lr_decay  # learning rate домножается на это число через каждые 1000 итераций
        self.messages = messages 
        # 1 - только результаты запусков и лучший результат
        # 2 - результаты оптимизации через 1000 итераций
        # 3 - рисунки для результатов каждого запуска
        self.tol = diam_tol  # отображаются диаметры, принадлежащие интервалу (tol*d_max,d_max)
        self.prec_opt = prec_opt  # множитель learning rate для "точной" оптимизации
        self.best_p = None
        self.best_poly = None
        self.best_d = almost_inf
        self.device = device

    def random_packing_torus(self):  # упаковка кругов в тор
        ####    U = torch.Tensor(self.poly.U)
        ####    v =  torch.Tensor(self.poly.v).reshape((len(self.poly.planes),1))
        x = torch.rand((self.n, self.d)).to(self.device)
        n1 = self.n * 3 ** self.d
        lr = 0.0003

        optimizer = torch.optim.Adam([x.requires_grad_()], lr=lr)
        mask = torch.tril(torch.ones(n1, n1, dtype=torch.bool), diagonal=-1).to(self.device)
        for i in tqdm.tqdm(range(1, self.n_iter1 + 1)):
            optimizer.zero_grad()
            xl = []
            for v in torch.tensor(list(product([-1, 0, 1], repeat=2)), device=self.device):
                xl.append(x + v)
            X = torch.cat(xl).to(self.device)

            x2 = torch.square(X)
            x2s = torch.sum(x2, 1)
            distm = -2 * X.mm(X.t()) + x2s + x2s.reshape((n1, 1))
            dist_points = 0.5 * torch.sqrt(torch.masked_select(distm, mask))
            # dist_lines =   U.mm(x.t()) + v
            y = -torch.min(dist_points)
            y.backward()
            optimizer.step()
            if i % messages_iter == 0:
                # lr = lr * 0.8
                for g in optimizer.param_groups:
                    g["lr"] = lr
                if self.messages >= 2:
                    print(i, " min_L = ", torch.min(dist_points), "  r = ", abs(float(y.detach().numpy())))
        return x.cpu().detach().numpy()

    def random_partition_torus(self, x0):  # оптимизация разбиения
        # self.od = OptDiagramNd(self.poly,x0)
        self.od = OptDiagramTorus(x0, device=self.device)
        # except:
        #     return almost_inf, None
        if not self.od.ok:
            return almost_inf, None
        x = torch.tensor(self.od.points).to(self.device)
        lr = self.lr_start
        x = x.requires_grad_()
        optimizer = torch.optim.Adam([x], lr=lr)
        yp = almost_inf
        y_best = almost_inf
        no_impr = 0
        precise = False
        for i in tqdm.tqdm(range(1, self.n_iter2 + 1)):
            # if i % 100 == 0:
            #     plot_partition(self)
            optimizer.zero_grad()
            y = self.od.forward(x)
            y.backward()
            optimizer.step()
            ycur = y.cpu().detach().numpy()
            if ycur < y_best:
                y_best = ycur
                no_impr = 0
            else:
                no_impr += 1
            if i % messages_iter == 0:
                if self.messages >= 1:
                    print(i, "   d = ", float(ycur))
                yp = y.detach().numpy()
            if no_impr > no_impr_steps:
                lr = lr * self.lr_decay
                if lr < min_lr:
                    break
                if (ycur - self.best_d) / lr > comp_heu:
                    break
                for g in optimizer.param_groups:
                    g["lr"] = lr
        d = self.od.forward(x.detach())
        self.od.points = x.cpu().detach().numpy()
        # d, dlist = self.od.diams(tol = self.tol)
        if self.messages >= 3:
            self.od.draw_poly(diams=dlist)
            plt.show()
        return d.cpu().detach().numpy(), self.od.points

    def multiple_runs(self, m):  # мультистарт, хранение лучшего разбиения
        # best_d = almost_inf
        for i in range(m):
            x = self.random_packing_torus()
            if np.isnan(x).any():
                if self.messages >= 1:
                    print("NaN in data")
                continue
            d, p = self.random_partition_torus(x)
            d = d ** 0.5
            new_rec = ""
            if self.od.polygons is None:
                continue
            true_d = find_true_diam(p, self.od.polygons)
            if not (p is None) and true_d < self.best_d:
                self.best_d = true_d
                self.best_p = p.copy()
                self.best_poly = self.od.polygons.copy()
                # self.best_err = self.od.err
                new_rec = "***"
            if self.messages >= 1:
                print("run {0}/{1}, d_max = {2}  {3}".format(i + 1, m, d, new_rec))
        self.od.points = self.best_p.copy()
        self.od.polygons = self.best_poly.copy()
        self.od.set_mask()
        # d, dlist = self.od.diams(tol = self.tol)
        # self.od.draw_poly(diams=dlist)
        # plt.show()
        # self.best_d = best_d
        return float(self.best_d)

    def from_file(self, filename):  # для тора не работает
        with open(filename, "r") as f:
            l = f.readlines()
            poly = json.loads(l[3])
            points = json.loads(l[4])
            self.od = OptDiagramNd(self.poly, saved=(poly, points))
            self.od.set_mask()
            # d, dlist = self.od.diams(tol = self.tol)
            self.best_d = d.detach().numpy()
            return self.best_d

    def to_file(self, filename):  # запись в файл
        # TODO: переделать в json
        with open(filename, "w") as f:
            f.write(str(self.n) + "\n")
            f.write(str(self.best_d) + "\n")  # максимальный диаметр
            f.write(str(len(self.od.points)) + "\n")  # число вершин разбиения
            json.dump(self.od.polygons, f)  # полигоны, списки вершин
            f.write("\n")
            json.dump(self.od.points.tolist(), f)  # вершины

    def to_string(self):
        res = ""
        res += str(self.n) + "\n"
        res += str(self.best_d) + "\n"
        res += str(len(self.od.points)) + "\n"  # число вершин разбиения
        res += json.dumps(self.od.polygons)  # полигоны, списки вершин
        res += "\n"
        res += json.dumps(self.od.points.tolist())  # вершины
        return res

    def diams(self, tol=0.01):  # для тора не работает
        l = self.best_d
        res = []
        for p in self.od.polygons:
            for i, j in combinations(p, 2):
                l1 = np.linalg.norm(self.od.points[i] - self.od.points[j])
                if 1.0 - tol < l1 / l < 1.0 + tol:
                    res.append((i, j))
        return res


def find_true_diam(pts, polygons):
    true_diam = 0
    if pts is None:
        return float("inf")
    for p in polygons:
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
    best_p = point.copy()
    min_dist = 10.0
    for v in product([-1.0, 0.0, 1.0], repeat=2):
        cur_p = point + np.array(v)
        dist = np.linalg.norm(cur_p - target)
        if dist < min_dist:
            min_dist = dist
            best_p = cur_p.copy()
    return best_p


def plot_partition(part, find_true_diam=True, plot=True):
    """
    Рисует разбиения тора так чтобы части выглядели нормально,
    находя нужные экземляры точек
    """
    if plot:
        fig = plt.figure(figsize=(10, 10))
    pts = part.od.points
    true_diam = 0
    for p in part.od.polygons:
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
        if find_true_diam:
            for i in range(len(pcur)):
                for j in range(i + 1, len(pcur)):
                    true_diam = max(true_diam, np.linalg.norm(pcur[i] - pcur[j]))
        if plot:
            midp = np.average(pcur, axis=0)
            plt.text(*midp, f"{len(pcur)}", fontsize=20)
            plt.fill(*zip(*pcur), alpha=0.4)
    if plot:
        draw_square()
        plt.axis("equal")
        # plt.xlim(-0.2, 1.2)
        # plt.ylim(-0.2, 1.2)
        print(f" n: {part.n}   diam: {part.best_d:.10}   true_diam: {true_diam:.10f}")
        plt.title(
            f" n: {part.n}   diam: {part.best_d:.10f}   true_diam: {true_diam:.10f}",
            fontsize=16,
        )
        plt.show()
    return true_diam
