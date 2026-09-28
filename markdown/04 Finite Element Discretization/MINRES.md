---
tags: [numerics, linear-algebra]
aliases: [Minimal residual method]
---

**MINRES** (Paige & Saunders 1975) solves $A\mathbf x = \mathbf b$ for a symmetric, possibly **indefinite or singular**, matrix. At step $k$ it minimises the residual norm over the Krylov space:

$$
\mathbf x_k = \arg\min_{\mathbf x\in\mathbf x_0+\mathcal K_k}\|\mathbf b-A\mathbf x\|_2 .
$$

It uses the same short Lanczos recurrence as CG, so its cost per iteration is similar, with slightly more vector operations.

**Compared with CG.**

- CG requires positive definiteness. MINRES only symmetry. It works for saddle-point systems, such as the Neumann problem with a Lagrange multiplier for the mean (see [[Null Space of the Neumann Problem]]).
- On singular *consistent* systems MINRES converges to a solution. With $\mathbf x_0 = 0$ it is the minimum-norm solution in exact arithmetic. On inconsistent singular systems it returns a least-squares solution; MINRES-QLP returns the minimum-norm one.
- The residual decreases monotonically, which makes stopping criteria reliable.
- It needs a symmetric positive definite preconditioner.

For pure-Neumann EIT forward and adjoint problems with mean-zero projection, MINRES with an [[Algebraic Multigrid]] preconditioner is a robust default.

**Block MINRES and symmetric scaling.** Krylov.jl provides block MINRES for $s$ right-hand sides. A MINRES preconditioner must be symmetric positive definite and is applied symmetrically. A diagonal (Jacobi) preconditioner can therefore be realised exactly by solving the scaled system
$$ \big(D^{-1/2}AD^{-1/2}\big)\,\mathbf y = D^{-1/2}\mathbf b,\qquad \mathbf x = D^{-1/2}\mathbf y,\qquad D = \operatorname{diag}(A). $$
Scaling preserves consistency for the singular Neumann matrix. The scaled null space is $D^{1/2}V$, and $D^{-1/2}\Pi\mathbf b$ is orthogonal to it, because $(D^{-1/2}\Pi\mathbf b)^\top D^{1/2}V = \mathbf b^\top\Pi V = 0$. The result is then [[Grounding of the Potential|grounded]] as for the other projected solvers (see [[Projected Conjugate Gradient]]). Blocks of linearly dependent right-hand sides are handled by solving for an orthonormal basis of their span and recombining.

## References

1. C. C. Paige, M. A. Saunders (1975). *Solution of Sparse Indefinite Systems of Linear Equations*. SIAM J. Numer. Anal. 12(4), 617–629. [doi:10.1137/0712047](https://doi.org/10.1137/0712047)
2. S.-C. T. Choi, C. C. Paige, M. A. Saunders (2011). *MINRES-QLP: A Krylov Subspace Method for Indefinite or Singular Symmetric Systems*. SIAM J. Sci. Comput. 33(4), 1810–1836. [doi:10.1137/100787921](https://doi.org/10.1137/100787921)
3. A. Montoison, D. Orban (2023). *Krylov.jl: A Julia basket of hand-picked Krylov methods*. J. Open Source Softw. 8(89), 5187. [doi:10.21105/joss.05187](https://doi.org/10.21105/joss.05187)
