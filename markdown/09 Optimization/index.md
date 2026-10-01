---
title: Optimization
tags: [overview, optimization]
---

Methods that minimise the regularised reconstruction functional: Newton-type methods for least squares, quasi-Newton methods, and proximal splitting for non-smooth and learned priors.

**Reading order.**

1. Least squares: [[Gauss-Newton Method]], [[Levenberg-Marquardt Method]], [[Line Search]].
2. Quasi-Newton: [[L-BFGS]], [[L-BFGS-B]] with [[Box Constraints on Conductivity]].
3. Splitting: [[Proximal Operator]], [[ADMM]], [[Chambolle-Pock Algorithm]], [[Nested ADMM Reconstruction]].
4. Stochastic methods: [[Stochastic and Adaptive Gradient Methods]].
5. When to stop: [[Stopping Criteria]], [[Noise-Level Stagnation Test]].

A guide to the choice of method is [[Choosing an Optimizer]].

**Related.** Gradients: [[07 Adjoint Gradients/index|Adjoint Gradients]]; regularisers: [[08 Regularization/index|Regularization]] and [[11 Learned Priors/index|Learned Priors]].

## References

1. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
