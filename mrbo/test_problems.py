from botorch.test_functions.synthetic import Ackley, Griewank, Levy, Rastrigin, Rosenbrock


def get_problem(name, dim):
    if name == "Ackley":
        return Ackley(dim=dim, negate=True)
    if name == "Rosenbrock":
        return Rosenbrock(dim=dim, negate=True)
    if name == "Rastrigin":
        return Rastrigin(dim=dim, negate=True)
    if name == "Griewank":
        return Griewank(dim=dim, negate=True)
    if name == "Levy":
        return Levy(dim=dim, negate=True)
    raise ValueError(f"Unknown problem: {name}")
