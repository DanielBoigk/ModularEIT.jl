---
title: Adjoint Gradients
tags: [overview, adjoint, optimization]
---

How the derivative of a reconstruction functional with respect to the conductivity is computed efficiently, independently of the number of unknowns, and how it is represented.

**Reading order.**

1. The setting: [[PDE-Constrained Optimization]], [[State Equation]], [[Lagrangian Formulation]], [[KKT Conditions]], with the tools of the [[Rules of the Calculus of Variations]].
2. The method: [[Adjoint State Method]], [[Adjoint Equation]], [[Functional Derivative of the Data Misfit]], and the [[Adjoint Method for the Dirichlet Problem]].
3. An alternative functional without adjoint: [[Kohn-Vogelius Functional]].
4. Representation and discretisation: [[Gradient Representation and the Riesz Map]], [[Discretize-then-Optimize vs Optimize-then-Discretize]], [[Automatic Differentiation vs Adjoint Methods]].
5. Practice: [[Gradient Testing]] and the [[Iterative Reconstruction Loop]].

**Next.** The gradients feed the methods of [[09 Optimization/index|Optimization]]; the overall picture is [[Anatomy of an EIT Reconstruction]].

## References

1. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
