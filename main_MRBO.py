import warnings

import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from joblib import Parallel, delayed
from test_problems import get_problem


warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)

CONFIG = {
    "problem_name": "Ackley",
    "dim": 5,
    "batch_size": 4,
    "n_init": 20,
    "n_iter": 20,
    "n_jobs": -1,
}


def select_one_candidate(i, sorted_indices, train_x, train_y, bounds, dim, batch_size):
    # The i-th candidate uses a different number of nearest samples.
    n_data = train_x.shape[0]
    n_local = int(n_data / batch_size * (i + 1))
    n_local = max(2 + i, min(n_local, n_data - (batch_size - 1 - i)))

    local_indices = sorted_indices[:n_local]
    local_x = train_x[local_indices]
    local_y = train_y[local_indices]

    # Build a local search box from the selected local samples.
    local_lower = torch.max(local_x.min(dim=0).values, bounds[0])
    local_upper = torch.min(local_x.max(dim=0).values, bounds[1])
    local_bounds = torch.stack([local_lower, local_upper])

    model = SingleTaskGP(
        train_X=local_x,
        train_Y=local_y,
        train_Yvar=torch.full_like(local_y, 1e-6),
        input_transform=Normalize(d=dim),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acq = ExpectedImprovement(model=model, best_f=local_y.max())
    candidate, _ = optimize_acqf(
        acq_function=acq,
        bounds=local_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
    )
    return candidate.squeeze(0)


if __name__ == "__main__":
    problem = get_problem(CONFIG["problem_name"], CONFIG["dim"])
    bounds = problem.bounds

    # Generate the initial design by Sobol sampling.
    sobol = torch.quasirandom.SobolEngine(dimension=CONFIG["dim"], scramble=True)
    train_x = unnormalize(sobol.draw(CONFIG["n_init"]), bounds)
    train_y = problem(train_x).unsqueeze(-1)

    for iteration in range(1, CONFIG["n_iter"] + 1):
        # Sort all samples according to their distance to the current best point.
        best_index = train_y.argmax()
        best_x = train_x[best_index]
        distances = torch.norm(train_x - best_x, dim=1)
        sorted_indices = torch.argsort(distances)

        # The local EI subproblems are independent and can be solved in parallel.
        candidates = Parallel(n_jobs=CONFIG["n_jobs"])(
            delayed(select_one_candidate)(
                i,
                sorted_indices,
                train_x,
                train_y,
                bounds,
                CONFIG["dim"],
                CONFIG["batch_size"],
            )
            for i in range(CONFIG["batch_size"])
        )
        new_x = torch.stack(candidates)
        new_y = problem(new_x).unsqueeze(-1)

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_value = -train_y.max().item()

        print(
            f"MRBO iter {iteration:02d}: "
            f"evals = {train_x.shape[0]}, best value = {best_value:.4f}"
        )
