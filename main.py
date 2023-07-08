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

    parts = [None] * 100

    n = args.n
    parts[n] = OptPartitionTorus(
        d=2,
        n=args.n,
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

    parts[n].multiple_runs(args.num_runs)
    print(f" n: {n} diam: {parts[n].best_diam:.8f}")
    
    if args.name_file_path is not None:
        parts[n].to_file(args.name_file_path)

    plot_partition(
        parts[n],
        diam_tolerance=args.diam_tolerance,
    )
