---
tags: [adjoint, gradient, software]
aliases: [Autodiff, AD]
---

There are two ways to get the gradient of $\sigma\mapsto J(L_\sigma^{-1}\mathbf g)$:

**1. Automatic differentiation (AD).** Differentiate through the code: assembly of $L_\sigma$, the linear solve, and the misfit. Reverse-mode AD (backpropagation) costs a small constant multiple of one forward evaluation, independent of the number of parameters.

- *Pros:* exact gradient of the discrete objective, including quadrature and projections. It is automatic for any differentiable metric or regulariser.
- *Cons:* differentiating *through* an iterative solver records every iteration, which uses a lot of memory and is inaccurate unless the solver has converged. It must therefore be combined with **implicit differentiation** of the linear solve: the adjoint of $\mathbf u = L^{-1}\mathbf g$ is $\bar{\mathbf g} = L^{-\top}\bar{\mathbf u}$ and $\bar L = -\bar{\mathbf g}\,\mathbf u^\top$. That rule is the [[Adjoint State Method]] at the matrix level. Differentiating sparse assembly and mutating code can also be technically fragile.

**2. Adjoint method, derived by hand.** Solve the [[Adjoint Equation]] explicitly and assemble $-\nabla u\cdot\nabla\lambda$. It reuses the forward solver and preconditioner, has minimal memory use, and is easy to batch over patterns.

The two approaches agree up to the difference between optimise-then-discretise and discretise-then-optimise (see [[Discretize-then-Optimize vs Optimize-then-Discretize]]). A common hybrid uses the hand-written adjoint for the PDE and AD for the misfit $\partial_u d$ and the regulariser $\nabla\mathcal R$.

Either way, verify gradients numerically (see [[Gradient Testing]]).

## References

1. A. Griewank, A. Walther (2008). *Evaluating Derivatives*, 2nd ed. SIAM. [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
2. W. S. Moses, V. Churavy (2020). *Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients*. NeurIPS 33. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2020/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf)
3. W. S. Moses et al. (2021). *Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme*. SC '21. [doi:10.1145/3458817.3476165](https://doi.org/10.1145/3458817.3476165)
