# The Python code of MRBO-SPACE

import warnings

import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

from benchmark import rosenbrock

warnings.filterwarnings("ignore")


# setting of the problem
dimension = 5
bounds = torch.tensor([[-5.0] * dimension, [10.0] * dimension], dtype=torch.double)

# the number of points selected in each iteration
batch_size = 4

# the number of initial design points
num_initial = 2 * dimension

# the number of iterations
num_iterations = 5

# initial design points using Latin hypercube sampling method
unit_x = qmc.LatinHypercube(d=dimension).random(num_initial)
train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.tensor(unit_x, dtype=torch.double)

# evaluating the initial design points with the real function
train_y = -rosenbrock(train_x).unsqueeze(1)

# print the current information to the screen
print(f"iteration 0: evaluations = {len(train_x)}, best = {-train_y.max().item():.6g}")


# the iteration
for iteration in range(1, num_iterations + 1):
    # the current best solution
    best_x = train_x[train_y.argmax()]
    candidates = []

    for i in range(batch_size):
        # build the local search region
        ratio = (i + 1) / batch_size
        local_lower = best_x + ratio * (bounds[0] - best_x)
        local_upper = best_x + ratio * (bounds[1] - best_x)
        local_bounds = torch.stack((local_lower, local_upper))

        # select the samples in the local search region
        inside = ((train_x >= local_lower) & (train_x <= local_upper)).all(dim=1)
        local_x = train_x[inside]
        local_y = train_y[inside]

        # build the Gaussian process model
        model = SingleTaskGP(
            local_x,
            local_y,
            train_Yvar=torch.full_like(local_y, 1e-6),
            input_transform=Normalize(d=dimension),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # maximize the expected improvement function
        ei = ExpectedImprovement(model, best_f=local_y.max())
        candidate, _ = optimize_acqf(
            ei,
            bounds=local_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
        )
        candidates.append(candidate.squeeze(0))

    # evaluating the candidate points with the real function
    new_x = torch.stack(candidates)
    new_y = -rosenbrock(new_x).unsqueeze(1)

    # add the new points to the design set
    train_x = torch.cat((train_x, new_x))
    train_y = torch.cat((train_y, new_y))

    # print the current information to the screen
    print(f"iteration {iteration}: evaluations = {len(train_x)}, "f"best = {-train_y.max().item():.6g}")


# print the best solution
best_index = train_y.argmax()
print("\nbest x:", train_x[best_index])
print("best y:", -train_y[best_index].item())
