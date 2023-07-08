import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        help="Number of partitions",
    )
    parser.add_argument("--n_start", type=int)
    parser.add_argument("--n_end", type=int)
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of optimization attempts (default: 10)",
    )
    parser.add_argument(
        "--name_file_path",
        type=str,
        help="Path to the file with the name"
    )
    parser.add_argument(
        "--save_file",
        type=bool,
        default=True,
        help="Whether to save to a file should be specified only in the case of n_start < n_end",
    )
    parser.add_argument(
        "--n_iter_circ",
        type=int,
        default=1000,
        help="Number of iterations for circular optimization (default: 1000)",
    )
    parser.add_argument(
        "--n_iter_part",
        type=int,
        default=5000,
        help="Number of iterations for partial optimization (default: 5000)",
    )
    parser.add_argument("--lr_start", type=float, default=0.005)
    parser.add_argument("--lr_decay", type=float, default=0.93)
    parser.add_argument("--precision_opt", type=float, default=0.001)
    parser.add_argument("--diam_tolerance", type=float, default=0.99)
    parser.add_argument("--messages", type=int, default=1)
    parser.add_argument("--may_plot", type=bool, default=False)

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()

    return args
