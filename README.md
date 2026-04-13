# Broadcasting Model

[![Tests](https://github.com/truemerrill/junction/actions/workflows/tests.yml/badge.svg)](https://github.com/truemerrill/junction/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://truemerrill.github.io/junction/)


This package provides a modeling and simulation framework for ion ensembles in electrode broadcasting architectures, where shared control waveforms generate a lattice of trapping potentials. The approach is based on a phase-space formulation in a co-moving frame, modeling transport with a time-dependent quadratic Hamiltonian. Small deviations from the ideal waveform are treated as stochastic perturbations, enabling efficient propagation of phase-space covariance.

## Slides

[Theory Slides](docs/slides.pdf)

## Key Features

- **Electrostatics + dynamics pipeline**  
  Construct local quadratic approximations and simulate ion motion using a 4th-order symplectic integrator under time-dependent control waveforms.

- **Phase-space covariance evolution**  
  Propagate uncertainty efficiently using a linearized stochastic model, avoiding expensive Monte Carlo sampling.

- **General noise modeling**  
  Support for arbitrary time-correlated (colored) noise sources.

- **Broadcasting ensemble framework**  
  Model transport across lattices of nominally identical wells, capturing statistical variation across the device.

- **JAX-based implementation**  
  High-performance, differentiable simulation using JAX and diffrax.

## Overview

Ion transport is modeled as a stochastic dynamical system:

- Dynamics are linearized about a moving equilibrium trajectory  
- Errors are represented as time-dependent forces acting on the ions  
- The system is expressed as an affine stochastic differential equation  
- Phase-space covariance is computed via convolution with the fundamental matrix solution  

This framework enables direct estimation of transport-induced heating and sensitivity to control errors without repeated trajectory sampling.

A full derivation of the model—including the co-moving frame Hamiltonian, normal mode transformation, and covariance propagation—is provided in the slides.


## Installation

We recommend that you use [`uv`](https://docs.astral.sh/uv/) by Astral to manage your Python environment.

```
uv venv venv --python=3.12
source venv/bin/activate
uv pip install -e ".[dev,docs]"
```

Once the environment is setup, you can run the unit tests with

```
pytest test
```