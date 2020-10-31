import argparse
from FashionMNIST.solver import Solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--show_n_images", type=int, default=12)
    parser.add_argument("--norm_data", type=bool, default=True)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=100)  # using mini-batches of 128 samples
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--ckpt_dir", type=str, default="./logs")

    args = parser.parse_args()

    solver = Solver(args)


if __name__ == "__main__":
    main()
