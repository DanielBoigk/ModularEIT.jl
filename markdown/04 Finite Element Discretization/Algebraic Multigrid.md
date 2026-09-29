---
tags: [numerics, linear-algebra]
aliases: [AMG]
---

**Algebraic multigrid (AMG)** is an optimal-complexity preconditioner for the SPD matrices of elliptic PDEs such as the [[Weighted Stiffness Matrix]].

**Idea.** Simple smoothers (Jacobi, Gauss–Seidel) quickly remove *oscillatory* error components but are slow on *smooth* ones. Multigrid transfers the smooth error to a coarser level, where it looks oscillatory again, and recurses. *Algebraic* multigrid builds the coarse levels from the matrix entries alone. Smoothed-aggregation or Ruge–Stüben coarsening uses strong couplings, so no mesh hierarchy is needed.

**Properties.**

- One V-cycle costs $\mathcal O(n)$. Used as a preconditioner for [[Conjugate Gradient Method|CG]] or [[MINRES]], the iteration count is nearly independent of the mesh size.
- It adapts to strongly varying coefficients $\gamma$, which geometric multigrid handles poorly, because coarsening follows the strong connections in $L_\gamma$.
- The setup (building the hierarchy) is the expensive part. In a reconstruction loop the hierarchy for a previous $\sigma$ can be reused as long as $\sigma$ changes little, since $L_\sigma$ is spectrally equivalent to $K$.
- For singular Neumann matrices, the coarsest-level solve must handle the constant null space (pseudo-inverse or projection).

On uniform rectangle grids, a transform-based preconditioner without σ-dependent setup is an alternative; see [[Fast Solvers on Rectangular Domains]].

**In ModularEIT.jl:** [`AMGPreconditioner`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.AMGPreconditioner).

## References

1. J. W. Ruge, K. Stüben (1987). *Algebraic Multigrid*. In: Multigrid Methods, SIAM Frontiers in Applied Mathematics, 73–130. [doi:10.1137/1.9781611971057.ch4](https://doi.org/10.1137/1.9781611971057.ch4)
2. P. Vaněk, J. Mandel, M. Brezina (1996). *Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems*. Computing 56, 179–196. [doi:10.1007/BF02238511](https://doi.org/10.1007/BF02238511)
