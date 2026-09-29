---
tags: [numerics, fem]
aliases: [K, Laplacian matrix]
---

The (unweighted) **stiffness matrix** discretises the Laplacian:

$$
K_{ij} = \int_\Omega\nabla\varphi_i\cdot\nabla\varphi_j\,\mathrm dx,\qquad \mathbf z^\top K\mathbf z = \|\nabla z_h\|^2_{L^2(\Omega)} .
$$

**Properties.**

- Symmetric positive **semi**definite. Its kernel consists of the constants, since $\nabla 1 = 0$ and $\sum_i\varphi_i\equiv1$ for Lagrange elements. It is singular without Dirichlet conditions (see [[Null Space of the Neumann Problem]]).
- Sparse: $K_{ij}\ne0$ only if $\varphi_i$ and $\varphi_j$ share a cell.
- Condition number $\mathcal O(h^{-2})$ on the complement of the kernel, so iterative solvers need preconditioning (see [[Algebraic Multigrid]]).
- Requires $H^1$-conforming elements ($P_1/Q_1$ or higher). For $P_0$ the gradient is zero inside cells.

**Uses.** $H^1$-seminorm [[Tikhonov Regularization]] ($\tfrac\beta2\mathbf z^\top K\mathbf z$), Laplacian smoothing, and as a Levenberg–Marquardt matrix $L_{\text{LM}}$ (see [[Levenberg-Marquardt Method]]). The conductivity-weighted version is the [[Weighted Stiffness Matrix]].

**In ModularEIT.jl:** [`assemble_stiffness!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.assemble_stiffness!), [`FEMatrices`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.FEMatrices).

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
