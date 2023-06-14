import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        help="Number of partitions",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    args = parser.parse_args()

    return args
