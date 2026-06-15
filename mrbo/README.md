# MRBO

This repository provides a simple demo implementation of Multi-Region Bayesian Optimization (MRBO).

MRBO is a batch Bayesian optimization method. At each iteration, it finds the current best point, sorts the observed samples by their distance to this point, and builds several local regions with different numbers of nearest samples. Each local region trains a Gaussian process model and selects one candidate by maximizing Expected Improvement. The selected candidates form a batch.

## Requirements

```bash
pip install -r requirements.txt
```

## Run

```bash
python main_MRBO.py
```

The default demo runs MRBO on the 5-D Ackley function and prints the best value at each iteration.

## Files

- `main_MRBO.py`: main MRBO demo script.
- `test_problems.py`: benchmark test functions used by the demo.
