---
title: Finite Elements
tags: [overview, fem]
---

The discretisation of the forward problem with finite elements: from the Galerkin principle to the matrices and operators that the solvers and the reconstruction work with.

**Reading order.**

1. The method: [[Galerkin Method]], [[Lagrange Finite Elements]], [[Numerical Quadrature and Assembly]].
2. The matrices: [[Mass Matrix]], [[Stiffness Matrix]], [[Weighted Stiffness Matrix]], and the [[Conductivity Tensor]] that makes the dependence on σ linear and explicit.
3. Boundary conditions and electrodes: [[Enforcing Dirichlet Conditions]], [[Discrete Electrode Models]], [[Boundary Mass and Stiffness Matrices]], [[Discrete Boundary Operator]], [[Discrete Fractional Sobolev Norms]].
4. The singular Neumann problem: [[Null Space of the Neumann Problem]], [[Grounding of the Potential]].
5. Moving functions between representations: [[L2 Projection]], [[Pixel Images and Finite Element Functions]].

A summary of the chain from the model to the linear systems is [[From Physics to Linear Algebra]].

**Next.** Solving the systems: [[05 Linear Solvers/index|Linear Solvers]]; choosing and adapting the mesh: [[06 Meshes and Geometry/index|Meshes and Geometry]].

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
