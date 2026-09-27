---
tags: [numerics, linear-algebra, performance]
aliases: [Batched solves, Multiple right-hand sides]
---

An EIT iteration solves the *same* matrix $L_\sigma$ against many right-hand sides: one state solve per current pattern and one adjoint solve per pattern,

$$
L_\sigma\,[\mathbf u_1\ \cdots\ \mathbf u_N] = [\mathbf g_1\ \cdots\ \mathbf g_N],\qquad
L_\sigma\,[\boldsymbol\lambda_1\ \cdots\ \boldsymbol\lambda_N] = [\mathbf r_1\ \cdots\ \mathbf r_N].
$$

**Batching strategies.**

- **Direct factorisation.** Factor $L_\sigma$ once (sparse Cholesky, or $LDL^\top$ after grounding) and do $2N$ triangular solves. This pays off when $N$ is large relative to the matrix size or high accuracy is needed. The factorisation must be redone whenever $\sigma$ changes.
- **Block Krylov methods** (block CG, block MINRES). They iterate on all right-hand sides together, sharing a larger Krylov space, and turn $N$ matrix–vector products into one sparse matrix–*matrix* product (SpMM). This uses memory bandwidth and caches much better and maps well to GPUs. Care is needed with rank deficiency (deflation) when right-hand sides become nearly dependent.
- **Independent parallel solves** of the individual systems, for example one per thread.

Since state and adjoint solves use the same matrix and preconditioner, both can be batched together once the residuals are known. Stacking the patterns also reduces the per-pattern overhead of assembling the gradient $-\nabla u_i\cdot\nabla\lambda_i$ (see [[Functional Derivative of the Data Misfit]]).

## References

1. D. P. O'Leary (1980). *The block conjugate gradient algorithm and related methods*. Linear Algebra Appl. 29, 293–322. [doi:10.1016/0024-3795(80)90247-5](https://doi.org/10.1016/0024-3795(80)90247-5)
2. Y. Saad (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. [doi:10.1137/1.9780898718003](https://doi.org/10.1137/1.9780898718003)
3. A. Montoison, D. Orban (2023). *Krylov.jl: A Julia basket of hand-picked Krylov methods*. J. Open Source Softw. 8(89), 5187. [doi:10.21105/joss.05187](https://doi.org/10.21105/joss.05187)
