---
tags: [overview, reconstruction]
---

An iterative EIT reconstruction repeatedly compares simulated with measured data and improves the conductivity. It combines five exchangeable pieces.

## 1. Data

Measurements are voltages for applied current patterns, or currents for applied voltages (see [[Measurement Protocols]], [[Current Patterns]]). Real data contain noise and modelling errors (see [[Noise Models for EIT Data]]). Simulated test data must be generated with a different model than the reconstruction to avoid the [[Inverse Crime]]. How much information the data carry is limited by the [[Decay of Boundary Measurements]].

## 2. Objective

A functional measures the disagreement between model and data:

- the least-squares misfit in a chosen metric (see [[Data Fidelity Terms]]);
- the [[Kohn-Vogelius Functional]], which compares a current-driven and a voltage-driven solution in the energy norm.

The problem is ill-posed (see [[Well-Posedness]], [[Stability of the Calderón Problem]]), so a regulariser is added, $J(\sigma) = \Phi(\sigma)+\alpha R(\sigma)$ (see [[Variational Regularization]]). $R$ can be classical (see [[Tikhonov Regularization]], [[Total Variation]]) or learned (see [[Classical and Learned Priors]]).

## 3. Gradient

The [[Adjoint State Method]] gives the gradient of the misfit at the cost of one additional linear solve, whatever the number of unknowns (see [[PDE-Constrained Optimization]], [[Lagrangian Formulation]]). The Jacobian (see [[Linearized EIT and the Sensitivity Kernel]]) costs one solve per measurement. The gradient is a dual vector; turning it into a direction requires a choice of inner product (see [[Gradient Representation and the Riesz Map]]). Gradients should always be checked against finite differences (see [[Gradient Testing]]).

## 4. Optimiser

Gauss–Newton-type methods exploit the least-squares structure. Quasi-Newton methods need only gradients. Proximal splitting methods handle non-smooth or learned regularisers (see [[Choosing an Optimizer]]). Bounds keep the conductivity positive (see [[Box Constraints on Conductivity]]). The iteration stops by the discrepancy principle or other rules (see [[Stopping Criteria]], [[Choosing the Regularization Parameter]]).

## 5. Linear algebra and discretisation

Each evaluation solves the forward problem and each gradient an adjoint problem, both with the same system matrix (see [[From Physics to Linear Algebra]], [[Choosing a Linear Solver]]). The mesh can be adapted to the current conductivity and to the quantity of interest (see [[Adaptive Meshing in EIT]]).

The loop of these steps is described in [[Iterative Reconstruction Loop]]. Direct, non-iterative alternatives are the [[D-bar Method]] and other [[Reconstruction Algorithms with Guarantees]].

## References

1. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
2. C. R. Vogel (2002). *Computational Methods for Inverse Problems*. SIAM. [doi:10.1137/1.9780898717570](https://doi.org/10.1137/1.9780898717570)
