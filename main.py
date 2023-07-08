import torch  # в коде на данный момент не реализованы вычисления на GPU
import numpy as np
import random
import json
import matplotlib.pyplot as plt

from arguments import get_args

from optClasses import OptPartitionTorus, plot_partition


if __name__ == "__main__":
    args = get_args()

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.n_start is None:
        assert args.n is not None, "Number of partitions is not defined"
        n_start = n_end = args.n
    else:
        assert args.n_start is not None and args.n_end is not None, \
            "Min and max number of partitions are not defined"
        n_start = args.n_start
        n_end = args.n_end

    parts = [None] * (n_end + 1)
    for n in range(n_start, n_end + 1):
        parts[n] = OptPartitionTorus(
            d=2,
            n=n,
            n_iter_circ=args.n_iter_circ,
            n_iter_part=args.n_iter_part,
            lr_start=args.lr_start,
            lr_decay=args.lr_decay,
            precision_opt=args.precision_opt,
            diam_tolerance=args.diam_tolerance,
            messages=args.messages,
            may_plot=args.may_plot,
            batch_size=args.batch_size,
            device=args.device,
        )

    for _ in range(args.num_runs):
        for n in range(n_start, n_end + 1):
            parts[n].multiple_runs(1)
            print(f" n: {n} diam: {parts[n].best_diam:.8f}")
            if args.save_file is not None:
                parts[n].save_to_file(f"jsons/tor_{n}.json")
                plot_partition(
                    parts[n],
                    diam_tolerance=args.diam_tolerance,
                    filename=f"pictures/tor_{n}.png"
                )
    
    if n_start == n_end and args.name_file_path is not None:
        parts[n].save_to_file(args.name_file_path)

    for n in range(n_start, n_end + 1):
        plot_partition(
            parts[n],
            diam_tolerance=args.diam_tolerance,
        )
