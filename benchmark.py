import torch


def rosenbrock(x):
    # the Rosenbrock function
    # xi = [-2.048,2.048]
    return torch.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, dim=1)
