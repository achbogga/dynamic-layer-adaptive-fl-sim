#!/usr/bin/env python3
"""Plot MNIST accuracy for FedAvg and DLA-AI."""

import argparse
import matplotlib.pyplot as plt

from dla_fl.end_to_end_mnist import fedavg, dla_ai


def main():
    p = argparse.ArgumentParser(description="Plot MNIST accuracies")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--out", default="docs/images/mnist_accuracy.png")
    args = p.parse_args()

    rounds = list(range(1, args.rounds + 1))
    fed_acc = fedavg(num_rounds=args.rounds)
    dla_acc = dla_ai(num_rounds=args.rounds)

    plt.plot(rounds, fed_acc, marker="o", label="FedAvg")
    plt.plot(rounds, dla_acc, marker="o", label="DLA-AI")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("MNIST Accuracy per Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()

