import torch 
import numpy as np
import json
import matplotlib.pyplot as plt
from shapely.ops import polygonize,unary_union
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiPoint, Point
from scipy.spatial import Voronoi
from itertools import combinations
import sys
from fractions import Fraction

from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull

pinf = 1E6

debug = False

def lineeq(n, c, x):
    return n.dot(x) - c

class CPoly:  # convex polygon
              
    def __init__(self, center=[0.5,0.5], pts = []):
        self.lines = []
        self.center = center
        self.points = pts.copy()
        # pts.append(pts[0])
        for i in range(len(pts) - 1):
            self.addline(pts[i],pts[i+1])
        self.addline(pts[0],pts[len(pts)-1])            

    def addline(self,a,b): 
        d = np.array(b) - np.array(a)
        n = np.array([-d[1],d[0]])
        n = n/np.linalg.norm(n)
        c = n.dot(a)
        if n.dot(self.center) - c < 0:
            n = -n
            c = -c
        self.lines.append([n, c])
        
    def mindist(self, p):
        m = pinf
        for l in self.lines:
            v = l[0].dot(p) - l[1]
            if v < m:
                m = v
        return m
    
    def prepare(self):
        n = len(self.lines)
        self.U = np.zeros([n, 2])
        self.v = np.zeros(n)
        k = 0
        for l in self.lines:
            self.U[k, :] = l[0]
            self.v[k] = l[1]
            k += 1
            
            
def voronoi_finite_polygons_2d(vor, radius=None):  # bounded Voronoi diagram
                                                  
    """
    https://gist.github.com/Sklavit/e05f0b61cb12ac781c93442fbea4fb55

    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 100 # distance to a point lying on the infinite edge
                                             

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
            
    
eps = 1E-4
eps2 = eps ** 2
from scipy.optimize import minimize


def eq_pts(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < eps2

def eq_pts_torus(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return abs(dx - round(dx)) < eps and abs(dy - round(dy)) < eps # close to an integer

class OptDiagram: # creating a diagram, computing the objective function
    def pt_index(self, p):
        for i in range(len(self.points)):
            if eq_pts(p, self.points[i]):
                return i
        return -1

    def set_mask(self):
        n = len(self.points)
        m = len(self.bound.lines)
        self.mask = np.zeros((n, n), dtype=bool)
        for p in self.polygons:
            for i, j in combinations(p[:-1], 2):
                self.mask[i, j] = True
             
        self.mask = torch.BoolTensor(self.mask) # 1 if i < j < n - 1 else 0                
        self.mask_lines = np.zeros((m, n - self.bound_vert_n), dtype=bool)
        self.line_link = []
        dist_lines = self.bound.U.dot(np.transpose(self.points)) - self.bound.v.reshape((m, 1))
        for i in range(dist_lines.shape[0]):
            for j in range(self.bound_vert_n, dist_lines.shape[1]):
                if abs(dist_lines[i, j]) < eps:
                    self.line_link.append([j, i])
                    self.mask_lines[i, j - self.bound_vert_n] = True
        self.mask_lines = torch.BoolTensor(self.mask_lines)

    def __init__(self, bound, points):
        self.bound = bound
        vor = Voronoi(points)
        self.vor = vor
        regions, vertices = voronoi_finite_polygons_2d(vor)

        min_x = vor.min_bound[0] - 0.1
        max_x = vor.max_bound[0] + 0.1
        min_y = vor.min_bound[1] - 0.1
        max_y = vor.max_bound[1] + 0.1

        mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
        bounded_vertices = np.max((vertices, mins), axis=0)
        maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
        bounded_vertices = np.min((bounded_vertices, maxs), axis=0)
        box = Polygon(bound.points)
        self.points = []
        for p in bound.points:
            self.points.append(p)
        self.bound_vert_n = len(self.points)
        self.penalty = 1e8
        self.decay = 2.718
        self.finetune_tol = 1e-9
        self.diam_coef = torch.Tensor([self.decay**(-i) for i in range(1000)]).double()

        self.polygons = []
        for region in regions:
            polygon = vertices[region]
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            cur = []
            for p in poly.exterior.coords:
                i = self.pt_index(p)
                if i == -1:
                    self.points.append(p)
                    cur.append(len(self.points) - 1)
                else:
                    cur.append(i) 
            if len(cur) > 0:
                self.polygons.append(cur)

        self.points = np.array(self.points)
        
        m = len(bound.lines)
        n = len(self.points)
        self.n = n
        self.set_mask()
        self.diams_mask = None
        self.bound_pt_tensor = torch.Tensor(bound.points)
        self.line_U = torch.tensor(bound.U)
        self.line_v = torch.tensor(bound.v.reshape((len(bound.lines),1)))

    def forward(self, x):
        x_all = torch.cat((self.bound_pt_tensor, x))
        n = len(x_all)
        x2 = torch.square(x_all)
        x2s = torch.sum(x2, dim=1)
        distm = -2 * x_all.mm(x_all.t()) + x2s + x2s.reshape((n, 1))
        diags = torch.sqrt(torch.masked_select(distm, self.mask))
        dist_lines = torch.abs(torch.masked_select(self.line_U.mm(x.t()) - self.line_v, self.mask_lines)) # distances to lines
        k = min(20, len(diags))
        Loss = self.diam_coef[:k].dot(torch.topk(diags, k).values) + 10.0*torch.sum(dist_lines)
        # print(torch.max(diags), 5.0 * torch.max(dist_lines))
        # print(torch.sum(torch.topk(diags, 10).values))
        # assert False
        return Loss
    
    def diams(self, tol=0.99): # set of approximate diameters
        x_all = torch.tensor(self.points)
        n = len(x_all)
        x2 = torch.square(x_all)
        x2s = torch.sum(x2, dim=1)
        distm = -2 * x_all.mm(x_all.t()) + x2s + x2s.reshape((n, 1))
        res = []
        diags = torch.sqrt(torch.masked_select(distm, self.mask))
        m = torch.max(diags)
        m2 = m * m
        for i in range(n):
            for j in range(n):
                if self.mask[i, j]:
                    if distm[i, j] > m2 * tol:
                        res.append([i, j])
        self.diams_list = res                
        self.diams_mask = np.zeros((n, n), dtype=bool)
        for p in res:
             self.diams_mask[p[0], p[1]] = True
        self.diams_mask = torch.BoolTensor(self.diams_mask) 
                        
        return m, res
    
    def diam_loss(self, x):
        x_all = torch.cat((self.bound_pt_tensor, x))
        n = len(x_all)
        x2 = torch.square(x_all)
        x2s = torch.sum(x2, dim=1)
        distm = -2 * x_all.mm(x_all.t()) + x2s + x2s.reshape((n, 1))
        diam2 = torch.masked_select(distm, self.diams_mask) 
        m = len(diam2)
        sum_diff = torch.sum(torch.abs(torch.zeros((m,m)) + diam2 - diam2.reshape((m,1))))
        dist_lines = torch.sum(torch.abs(torch.masked_select(self.line_U.mm(x.t()) - self.line_v, self.mask_lines)))
        return 1e-2*torch.max(diam2) + sum_diff * self.penalty + dist_lines * self.penalty

    
    def diam_loss_simple(self, x, d):
        x_all = torch.cat((self.bound_pt_tensor, x))
        n = len(x_all)
        x2 = torch.square(x_all)
        x2s = torch.sum(x2, dim=1)
        distm = -2 * x_all.mm(x_all.t()) + x2s + x2s.reshape((n, 1))
        diam2_diff = torch.sum(torch.square((torch.masked_select(distm, self.diams_mask) - d*d)*1000.0))
        dist_lines = torch.sum(torch.square((torch.masked_select(self.line_U.mm(x.t()) - self.line_v, self.mask_lines))*1000.0))
        return diam2_diff + dist_lines
    
    

    def draw_poly(self, x = None, diams = None):
        fig,ax = plt.subplots(figsize=(8,8))

       
#        plt.figure()
        if x is None:
            points = self.points
        else:
            points = np.concatenate((self.bound_pt_tensor.numpy(), x))
        for p in self.polygons:
            cur = []
            for v in p:
                cur.append(points[v])
            xy = np.array(cur)    
            p = plt.Polygon(xy, fill=False) 
            ax.add_patch(p)   
            #plt.fill(*zip(*cur), alpha=0.4)
        if not diams is None:
            for d in diams:
                p1 = points[d[0]]
                p2 = points[d[1]]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='dotted', color = 'k')
        plt.axis('equal')
        plt.axis('off')
        plt.xlim(min(points[:, 0]) - 0.1, max(points[:, 0]) + 0.1)
        plt.ylim(min(points[:, 1]) - 0.1, max(points[:, 1]) + 0.1)

messages_iter = 1000
almost_inf = 100.0

class OptPartition():   # search for the optimal partition

    def __init__(self, points, n, n_iter_circ = 2000, n_iter_part = 70000, lr_start = 0.01, lr_decay = 0.99, prec_opt = 1/1000, diam_tol = 0.99, messages = 1):
        self.n_iter1 = n_iter_circ  # max iter for the sphere packing
        self.n_iter2 = n_iter_part  # max iter for the partition tuning
        self.n = n          # number of parts
        self.poly = CPoly(center = [0.5,0.5], pts = points)
        self.poly.prepare()
        self.U = torch.Tensor(self.poly.U)
        self.v = torch.Tensor(self.poly.v).reshape((len(self.poly.lines),1))
        self.lr_start = lr_start   # initial value of the learning rate
        self.lr_decay = lr_decay   # value by which learning rate is multiplied each 1000 iterations
        self.messages = messages   # 1 - only final results, 2 - each 1000 iterations, 3 - plot diagrams for the each restart
        self.tol = diam_tol      # diameters to be displayed (tol*d_max,d_max)
        self.prec_opt = prec_opt   # multiplier for the learning rate, 'high precision'
        self.best_d = almost_inf

    def random_packing(self):  # circle packing
        U = torch.Tensor(self.poly.U)
        v = torch.Tensor(self.poly.v).reshape((len(self.poly.lines),1))
        x = torch.rand((self.n, 2), requires_grad=True) 
        optimizer = torch.optim.Adam([x], lr=0.003)
        mask = torch.BoolTensor([[i < j for i in range(self.n)] for j in range(self.n)])
        for i in range(self.n_iter1):
            optimizer.zero_grad()
            x2 = torch.square(x)
            x2s = torch.sum(x2, dim=1)
            distm = -2 * x.mm(x.t()) + x2s + x2s.reshape((self.n, 1))
            dist_points = 0.5 * torch.sqrt(torch.masked_select(distm, mask))
            dist_lines = U.mm(x.t()) - v
            y = -torch.min(torch.min(dist_points), torch.min(dist_lines)) 
            y.backward(retain_graph=True)
            optimizer.step()
            if (i + 1) % messages_iter == 0:
                if self.messages >= 2:
                    print(i + 1,'  r = ', abs(float(y.detach().numpy())))
        return x.detach().numpy()

    def random_partition(self, x0): # partition optimization
        self.od = OptDiagram(self.poly, x0)
        t = np.array(self.od.points[self.od.bound_vert_n:])
        x = torch.tensor(t, requires_grad=True)
        lr = self.lr_start
        optimizer = torch.optim.Adam([x], lr=lr)
        yp = almost_inf
        precise = False
        precise_cnt = 0
        for i in range(self.n_iter2):
            optimizer.zero_grad()
            y = self.od.forward(x)
            y.backward(retain_graph=True)
            optimizer.step()
            if (i + 1) % messages_iter == 0:
                if self.messages >= 2:
                    print(i + 1, ' d = ', float(y.detach().numpy()))
                if self.messages >= 3:
                    d, dlist = self.od.diams(tol=self.tol)
                    self.od.draw_poly(diams=dlist)
                    plt.show()
                if y.detach().numpy() > yp:
                    if precise: 
                        precise_cnt +=1
                        if precise_cnt > 15:
                          break
                        if yp * 1.01 > self.best_d:
                          break  
                    else:
                        lr = lr * self.prec_opt
                        precise = True
                yp = y.detach().numpy()
                lr = lr * self.lr_decay
                for g in optimizer.param_groups:
                    g['lr'] = lr
        self.od.points[self.od.bound_vert_n:] = x[:, :].detach().numpy()
        d, dlist = self.od.diams(tol=self.tol)
        if self.messages >= 3:   
            self.od.draw_poly(diams=dlist)
            plt.show()
        return d.detach().numpy(), self.od.points

    def finetune_binsearch(self):
        d0 = float(self.best_d)
        tol = 1e-7 #self.od.finetune_tol
        tol1 = 1e-14
        x0 = np.array(self.od.points[self.od.bound_vert_n:]).copy()
        n = len(x0)
        x0 = x0.reshape((n*2,))
        def func(x, d):
            xt = torch.Tensor(x.reshape((n,2))).double()
            return self.od.diam_loss_simple(xt,d).detach().numpy()
        
        mstr = 'Nelder-Mead'
        d = d0
        calls = 1
        res = minimize(func, x0, args = (d,), method = mstr, tol = tol1)
        print(d, res.fun)
        if res.fun > tol:
            dmin = d
            while True:                
                d += 1e-4
                if d > 2.0:
                    return None
                res = minimize(func, x0, args = (d,), method = mstr, tol = tol1)
                calls += 1
                print(d, res.fun)
                if res.fun < tol:
                    dmax = d
                    break
        else:   
            dmax = d
            while True:                
                d -= 1e-4
                if d < 0.0:
                    return None
                res = minimize(func, x0, args = (d,), method = mstr, tol = tol1)
                print(d, res.fun)
                calls += 1
                if res.fun > tol:
                    dmin = d
                    break
        while True:
            print(f'[{dmin}, {dmax}]')
            d = (dmin + dmax)/2
            res = minimize(func, x0, args = (d,), method = mstr, tol = tol1)
            print(d, res.fun)
            calls += 1
            if res.fun > tol:
                dmin = d
            else:
                dmax = d
            if dmax - dmin < 1e-8:
                break
        res = minimize(func, x0, args = (dmax,), method = mstr, tol = tol1)
        print(d, res.fun)
        calls += 1
        self.od.points[self.od.bound_vert_n:] = res.x.reshape((n,2))#x[:, :]
        d, dlist = self.od.diams(tol=self.tol)
        if self.messages >= 3:   
            self.od.draw_poly(diams=dlist)
            plt.show()
        return d.detach().numpy(), self.od.points, calls

                
    def multiple_runs(self, m, sufficient_diam=0): # multistart 
        self.best_d = almost_inf
        for i in range(m):
            while True:
              x = self.random_packing()
              if not np.isnan(np.sum(x)):
                break
            d, p = self.random_partition(x)
            new_rec = ''
            if d < self.best_d:
                self.best_d = d
                self.best_p = p.copy()
                self.best_poly = self.od.polygons.copy()
                new_rec = '***'
                if self.messages >= 1:
                    print(f" run {i + 1}/{m} d_max = {d:.8f} {new_rec}")
            if self.best_d <= sufficient_diam:
                break
        self.od.points = self.best_p.copy()
        self.od.polygons = self.best_poly
        self.od.set_mask()
        d, dlist = self.od.diams(tol = self.tol)                                
        if self.messages >= 1:
            if debug:
                print('best_p', self.best_p.shape, self.best_p)
                print('od.points', self.od.points.shape, self.od.points)
            self.od.draw_poly(diams=dlist)
            plt.title(f' n: {self.n} d: {self.best_d}')
            plt.show()
        return float(self.best_d)

    def to_file(self,filename): 
        with open(filename,'w') as f:
            f.write(str(self.n)+'\n') 
            f.write(str(self.best_d)+'\n') 
            f.write(str(len(self.od.points))+'\n') 
            json.dump(self.od.polygons,f) 
            f.write('\n')
            json.dump(self.od.points.tolist(),f) 

            

def square():
    return [[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]]

def triangle(): 
    sq2 = 2**0.5/4
    sq6 = 6**0.5/4
    return [[sq2,-sq2],[sq6,sq6],[-sq2,sq2]]
            

from numpy.random import choice
import glob
import sys


arg = sys.argv
if len(arg) == 0 or arg[1]=='sq':
    shape = square()
    prefix = 'sq'
    print('searching for optimal partitions of the square...')
else:
    if arg[1] == 'tr':
        shape = triangle()
        prefix = 'tr'
        print('searching for optimal partitions of the equilateral triangle...')
    else:
        print(f'unknown shape: {arg[1]}')
        sys.exit(1)


best_results = dict()

files = glob.glob(f'res/{prefix}*.txt')
for fname in files:
    with open(fname, 'r') as f:
        L = f.readlines()
        n = int(L[0])
        d = float(L[1])
        best_results[n] = d

if len(best_results.keys())>0:
    print('current records:')
    print(best_results)        
        
n_range = list(range(6,50))
it_circ_range = list(range(100,3000))

if len(arg)>2:    
    runs = int(arg[2])
else:
    runs = 5

n = min(n_range)

while True:
    ic = choice(it_circ_range)
    op = OptPartition(shape, n,  n_iter_circ = ic, lr_start = 1E-2, lr_decay=0.98, messages = 0)
    op.multiple_runs(runs, sufficient_diam = 0.0)
    v =  best_results.get(n, False)
    if not v or op.best_d < v:        
        print(f'new record for k = {n}: {op.best_d}')
        best_results[n] = op.best_d
        op.to_file(f'res/{prefix}{n}.txt')
    n += 1
    if not n in n_range:
        n = min(n_range)
