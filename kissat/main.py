import os

import networkx as nx
import subprocess as sp
from itertools import product
from numba import jit, njit
import numpy as np

import argparse
import time


def read_coloring(filename, n_colors):
    res = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == 'v':
                cur = line.strip().split()
                for i in range(1, len(cur)):
                    if cur[i][0] != '-' and cur[i] != '0':
                        lit = int(cur[i])
                        k = (lit - 1) // n_colors
                        if k not in res:
                            res[k] = (lit - 1) % n_colors
    return res


def write_cnf(g, filename, size, n_colors, init=[]):
    with open(filename, 'w') as f:
        num_variables = g.number_of_nodes() * n_colors
        num_disjuncts = g.number_of_nodes() + g.number_of_edges() * n_colors + len(init)
        f.write(f"p cnf {num_variables} {num_disjuncts}\n")

        for u, v in g.edges():
            for c in range(1, n_colors + 1):
                f.write(f"-{(u * n_colors + c)} -{(v * n_colors + c)} 0\n")

        for i in g.nodes():
            L = i * n_colors + 1
            R = (i + 1) * n_colors + 1
            f.write(' '.join(map(str, range(L, R))) + ' 0\n')

        for ver, color in init:
            f.write(str(ver * n_colors + color) + ' 0\n')


def write_to_file(dcol, filename, grid_size):
    with open(filename, 'w') as file:
        for i, color in enumerate(dcol.values()):
            file.write(str(color))
            if (i + 1) % grid_size == 0:
                file.write('\n')
        file.write('\n')


def run_kissat(cnf_file, timer, n_colors, output_file, col_file, grid_size):
    cmd = f"timeout {timer} ./kissat1 {cnf_file} > {col_file}"
    sp.call(cmd, shell=True)
    # !timeout $timer ./kissat1 test.cnf > col.txt
    ans = str(sp.run(f"""grep "exit **" {col_file}""", shell=True, capture_output=True).stdout)
    dcol = read_coloring(col_file, n_colors)
    if not ans:
        res = 'timeout'
    elif '20' in ans:
        res = False
    elif '10' in ans:
        write_to_file(dcol, output_file, grid_size)
        res = True
    else:
        res = 'unknown'
    return res, dcol


def coloring(g, n_colors, size, timer):
    write_cnf(g, 'test.cnf', size, n_colors)
    res, dcol = run_kissat('test.cnf', timer, n_colors)
    return res


def init_circles_five(grid_size, radius=0):
    vector = np.array([2, 1]) / 5
    init_res = []
    for i in range(5):
        cur_color = i + 1
        cur_center = i * vector % 1
        for x in range(grid_size):
            for y in range(grid_size):
                cur_point = np.array([x, y]) / grid_size
                if np.linalg.norm(cur_point - cur_center) < radius:
                    init_res.append((x * grid_size + y + 1, cur_color))
    return init_res


def torus_graph(n, r2):
    sh = []
    for p in np.array(list(product(list(range(n // 2 + 1)), repeat=2))):
        if (p ** 2).sum() >= r2:
            sign = np.array([-1, 1])
            sh.append(p)
            sh.append(-p)
            sh.append(p * sign)
            sh.append(p * -sign)
    g = nx.Graph()
    to_vertex = lambda p: p[0] * n + p[1]
    for p in np.array(list(product(list(range(n)), repeat=2))):
        for q in sh:
            g.add_edge(
                to_vertex(p),
                to_vertex((p + q) % n)
            )
    return g


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_colors", type=int)
    parser.add_argument("--grid_start", type=int)
    parser.add_argument("--grid_end", type=int, default=100500)
    parser.add_argument("--grid_step", type=int)
    parser.add_argument("--percent_dev", type=float, default=1.0)
    parser.add_argument("--diam", type=float)
    parser.add_argument("--timer", type=int, default=3600)
    parser.add_argument("--init_radius", type=float, default=None)  # used only for five parts
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)
    np.random.seed(args.seed)

    for grid_size in range(args.grid_start, args.grid_end, args.grid_step):

        timer = args.timer
        left = int((args.diam * ((100 - args.percent_dev) / 100) * grid_size) ** 2)
        right = int((args.diam * grid_size) ** 2) + 10
        print(f"grid_size: {grid_size}  left: {left}  right: {right}")

        while right - left > 1:
            square_diameter = (left + right) // 2
            g = torus_graph(grid_size, square_diameter)

            if args.init_radius is not None:
                init_res = init_circles_five(grid_size, radius=args.radius)
            else:
                init_res = []
            
            cnf_name = f'cnfs/{args.n_colors}_test_{grid_size}_{square_diameter}_seed_{args.seed}.cnf'
            write_cnf(g, cnf_name, size=0, n_colors=args.n_colors, init=init_res)

            start_time = time.time()
            res, dcol = run_kissat(
                cnf_file=cnf_name,
                timer=timer,
                n_colors=args.n_colors,
                output_file=f"logs/{args.n_colors}_output_n={grid_size}_d2={square_diameter}.txt",
                col_file=f"cols/{args.n_colors}_col_n={grid_size}_d2={square_diameter}.txt",
                grid_size=grid_size,
            )
            end_time = time.time()

            print(f" time: {end_time - start_time:.3f}  ", end='')
            print(f" grid_size: {grid_size}  square_diameter: {square_diameter}  res: {res}")
            if res == True:
                right = square_diameter
            elif res == False:
                left = square_diameter
            else:
                timer = timer * 2
                print("Doubling the timer: ", timer)
        print("Summary")
        print(f" left: {left}  right: {right}  ")
        print(f" lower diameter estimate: {left ** 0.5 / grid_size:.6f}")
        print()