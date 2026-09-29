---
tags: [optimization]
aliases: [Gauss-Newton, GN]
---

For a nonlinear least-squares objective $\Phi(\sigma) = \tfrac12\|r(\sigma)\|^2$ with residual $r(\sigma) = \mathcal F(\sigma)-y$ and Jacobian $J = r'(\sigma)$:

- gradient: $\nabla\Phi = J^\top r$;
- Hessian: $\nabla^2\Phi = J^\top J + \sum_k r_k\nabla^2r_k$.

**Gauss–Newton** drops the second-order term, which is small near a good fit, and solves at each step the linearised least-squares problem

$$
\min_\delta\|r+J\delta\|^2\quad\Longleftrightarrow\quad J^\top J\,\delta = -J^\top r ,
$$

then updates $\sigma\leftarrow\sigma+\tau\delta$ with a step length $\tau$ from a [[Line Search]].

**Jacobian in EIT.** Row $(i,j)$ of $J$ is the derivative of the $j$-th boundary measurement of pattern $i$. By the [[Linearized EIT and the Sensitivity Kernel|linearisation identity]] its entries are integrals of $-\nabla u_i\cdot\nabla v_j$, where $v_j$ is the solution driven by the $j$-th measurement functional. Building $J$ explicitly costs one extra solve per measurement. Matrix-free variants only need products $J\delta$ (linearised forward) and $J^\top w$ (adjoint solves), and solve the normal equations with [[Conjugate Gradient Method|CG]] or [[LSQR]].

If only per-pattern gradients $\nabla J_i$ are available (one adjoint solve per pattern, see [[Adjoint State Method]]), a cheaper reduced Gauss–Newton model treats each pattern's misfit as one scalar residual $\rho_i = \sqrt{2J_i}$, with gradient $\nabla\rho_i = \nabla J_i/\rho_i$. Since $\sum_i J_i = \tfrac12\sum_i\rho_i^2$, Gauss–Newton applies with the $N\times n$ Jacobian whose rows are $\nabla\rho_i^\top$. It combines the $N$ pattern gradients into one search direction, at the price of a coarser curvature model than the full measurement Jacobian.

**Properties.** Locally quadratic convergence for zero-residual problems and linear convergence for small residuals. The matrix $J^\top J$ is extremely ill-conditioned in EIT, so the step must be damped, which gives the [[Levenberg-Marquardt Method]]. With a regulariser $\beta\mathcal R$ one solves $(J^\top J+\beta\nabla^2\mathcal R)\delta = -(J^\top r+\beta\nabla\mathcal R)$.

**In ModularEIT.jl:** [`GaussNewton`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.GaussNewton), [`residual_and_jacobian!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/objectives/#ModularEIT.residual_and_jacobian!).

## References

1. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed., Ch. 10. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
2. W. R. B. Lionheart (2004). *EIT reconstruction algorithms: pitfalls, challenges and recent developments*. Physiol. Meas. 25(1), 125–142. [doi:10.1088/0967-3334/25/1/021](https://doi.org/10.1088/0967-3334/25/1/021)
