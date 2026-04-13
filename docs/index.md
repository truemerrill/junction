# Broadcasting Model

This package provides a modeling and simulation framework for ion ensembles in electrode broadcasting architectures, where shared control waveforms generate a lattice of trapping potentials. The core approach uses a phase-space formulation in a co-moving frame, modeling transport with a time-dependent quadratic Hamiltonian. Small deviations from the ideal waveform are treated as stochastic perturbations, enabling efficient propagation of phase-space covariance through the system.


## [Slides](slides.pdf)


## Key Features

- **Electrostatics + dynamics pipeline**  
  Use local quadratic approximations and simulate ion dynamics with a 4th-order symplectic integrator under time-dependent control waveforms.

- **Phase-space covariance evolution**  
  Efficiently propagate uncertainty using a linearized stochastic model, avoiding expensive Monte Carlo sampling.

- **General noise modeling**  
  Supports arbitrary time-correlated (colored) noise sources acting on the ions.

- **Broadcasting ensemble framework**  
  Models transport across lattices of nominally identical wells, capturing statistical variation across the device.

- **JAX-based implementation**  
  High-performance, differentiable simulation using JAX and diffrax.

## Overview

The model treats ion transport as a stochastic dynamical system:
- Ion motion is linearized about a moving equilibrium trajectory
- Errors are represented as time-dependent forces acting on the system
- The resulting dynamics can be expressed as an affine stochastic system
- The phase-space covariance is computed via convolution with the fundamental matrix solution

This enables direct estimation of transport-induced heating and sensitivity to control errors without repeated trajectory sampling.

A detailed derivation of the model, including the co-moving frame Hamiltonian, normal mode transformation, and covariance propagation, is provided in the slides.

## Use Cases

- Estimating transport-induced heating
- Sensitivity analysis of waveform errors
- Modeling hardware-induced distortions (e.g., filters, DACs)
- Design and optimization of transport waveforms