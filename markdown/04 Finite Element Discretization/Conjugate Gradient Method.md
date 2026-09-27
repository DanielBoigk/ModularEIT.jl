---
tags: [numerics, linear-algebra]
aliases: [CG]
---

The **conjugate gradient (CG) method** solves $A\mathbf x = \mathbf b$ for a symmetric positive definite (SPD) matrix $A$. At step $k$ it minimises the energy norm of the error over the Krylov space:

$$
\mathbf x_k = \arg\min_{\mathbf x\in\mathbf x_0+\mathcal K_k}\|\mathbf x-\mathbf x_*\|_A,\qquad \mathcal K_k = \operatorname{span}\{\mathbf r_0, A\mathbf r_0,\dots,A^{k-1}\mathbf r_0\}.
$$

Each iteration needs one matrix–vector product and a few vector operations. Memory use is constant.

**Convergence.**
$$\|\mathbf x_k-\mathbf x_*\|_A\le2\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^k\|\mathbf x_0-\mathbf x_*\|_A,\qquad\kappa = \operatorname{cond}(A).$$
For FEM stiffness matrices $\kappa = \mathcal O(h^{-2})$, so a preconditioner is essential (see [[Algebraic Multigrid]]). Preconditioned CG needs an SPD preconditioner.

**In EIT.**

- *Forward and adjoint solves* with $L_\sigma$ are SPD after grounding, or consistent semidefinite systems with mean-zero data (see [[Null Space of the Neumann Problem]]).
- *Gauss–Newton normal equations* $(J^\top J+\lambda L_{\text{LM}})\delta = -J^\top r$ are SPD. CG only needs products with $J$ and $J^\top$ (see [[Gauss-Newton Method]]).
- *Tikhonov prox*: $(\beta K+\rho I)z = \rho y$ (see [[Tikhonov Regularization]]).

For symmetric indefinite or singular systems use [[MINRES]]. For least-squares problems with rectangular matrices use [[LSQR]].

## References

1. M. R. Hestenes, E. Stiefel (1952). *Methods of conjugate gradients for solving linear systems*. J. Res. Natl. Bur. Stand. 49(6), 409–436. [doi:10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)
2. Y. Saad (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. [doi:10.1137/1.9780898718003](https://doi.org/10.1137/1.9780898718003)
