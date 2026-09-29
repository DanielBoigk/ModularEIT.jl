---
tags: [overview, forward-problem, fem]
---

EIT starts with Maxwell's equations and ends, in the computer, with sparse symmetric linear systems. This article follows the chain of modelling and discretisation steps between the two.

## 1. The physical model

Under the [[Quasi-Static Approximation]], [[Ohm's Law in Continuum Form]] together with the [[Continuity Equation]] gives the [[Conductivity Equation]]

$$
\nabla\cdot\sigma\,\nabla u = 0\quad\text{in }\Omega .
$$

For alternating currents the conductivity becomes complex (see [[Complex Conductivity]]). How current enters through the electrodes is modelled by an electrode model, from the idealised continuum model to the [[Complete Electrode Model]] (see [[Electrode Models]]).

## 2. The analytic problem

Prescribing currents gives a [[Neumann Problem]], prescribing voltages a [[Dirichlet Problem]]. Both are well posed in the right function spaces (see [[Sobolev and Trace Spaces]]). The Neumann problem determines the potential only up to a constant, which is fixed by a grounding condition. The [[Weak Formulation of the Conductivity Equation]] and the [[Lax-Milgram Theorem]] give existence and uniqueness. The [[Dirichlet and Thomson Principles]] characterise the solution by energy minimisation. The map from applied currents to measured voltages is the [[Neumann-to-Dirichlet Map]]. Sampled at the electrodes, it is the [[Forward Map]].

## 3. Discretisation

The [[Galerkin Method]] restricts the weak form to a finite element space, e.g. [[Lagrange Finite Elements]] for the potential and piecewise constants for the conductivity. The integrals become matrices:

- the [[Weighted Stiffness Matrix]] $L(\sigma) = \int\sigma\,\nabla\varphi_i\cdot\nabla\varphi_j$, linear in σ, which the [[Conductivity Tensor]] makes explicit;
- the [[Mass Matrix]], [[Stiffness Matrix]] and [[Boundary Mass and Stiffness Matrices]] for norms, projections and boundary terms;
- the injection and measurement operators of the [[Discrete Electrode Models]].

The singular Neumann system needs its null space handled explicitly (see [[Null Space of the Neumann Problem]], [[Grounding of the Potential]]). Dirichlet data are imposed by elimination (see [[Enforcing Dirichlet Conditions]]).

## 4. The linear systems

Every forward solve, every adjoint solve and every Jacobian evaluation solves

$$
A(\sigma)\,X = B
$$

with a sparse, symmetric, positive semidefinite $A(\sigma)$ and one column per current pattern. Which solver fits depends on the mesh, the number of right-hand sides and how often σ changes (see [[Choosing a Linear Solver]]). How fine the mesh has to be, and where, is the subject of [[06 Meshes and Geometry/index|Meshes and Geometry]].

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Rev. 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
