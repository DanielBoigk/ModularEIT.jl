---
tags: [numerics, fem]
aliases: [M]
---

The **mass matrix** is the Gram matrix of the basis in $L^2(\Omega)$:

$$
M_{ij} = \int_\Omega\varphi_i\,\varphi_j\,\mathrm dx .
$$

For $u_h = \sum u_i\varphi_i$ and $v_h = \sum v_i\varphi_i$: $(u_h,v_h)_{L^2} = \mathbf u^\top M\mathbf v$. So $\|u_h\|_{L^2}^2 = \mathbf u^\top M\mathbf u$. The Euclidean norm of the coefficient vector is *not* the $L^2$ norm of the function, except up to scaling on uniform meshes.

**Properties.** Symmetric positive definite and sparse, with condition number $\mathcal O(1)$ independent of $h$ on quasi-uniform meshes. For $P_0$ elements it is diagonal with the cell areas.

**Assembly.** A sum over cells of local matrices $M_e[i,j] = \sum_q \varphi_i(x_q)\varphi_j(x_q)\,w_q\,|\det J_e(x_q)|$ (see [[Numerical Quadrature and Assembly]]).

**Uses.**

- $L^2$ inner products and norms of discrete functions, for example in $L^2$-[[Tikhonov Regularization]];
- the [[L2 Projection]] $M\mathbf z = \mathbf b$ of non-polynomial functions such as the adjoint gradient $-\nabla u\cdot\nabla\lambda$;
- turning a gradient vector (a functional, entries $\partial J/\partial z_i$) into a function: the $L^2$ Riesz representer is $M^{-1}\nabla_z J$ (see [[Gradient Representation and the Riesz Map]]).

Since $M$ is fixed for a given mesh, a sparse Cholesky factorisation is computed once and reused.

**In ModularEIT.jl:** [`assemble_mass!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEITFerrite.assemble_mass!), [`FEMatrices`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.FEMatrices).

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. A. Ern, J.-L. Guermond (2004). *Theory and Practice of Finite Elements*. Springer. [doi:10.1007/978-1-4757-4355-5](https://doi.org/10.1007/978-1-4757-4355-5)
