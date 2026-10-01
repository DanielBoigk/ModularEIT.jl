---
tags: [numerics, fem]
aliases: [Q1 elements, P1 elements, Quadrilateral elements, P0 elements]
---

**Lagrange elements** use piecewise polynomial basis functions defined by their values at nodes. $\varphi_i(x_j) = \delta_{ij}$, so the coefficient vector is the vector of nodal values.

- $P_k$ on triangles/tetrahedra: complete polynomials of degree $\le k$.
- $Q_k$ on quadrilaterals/hexahedra: tensor-product polynomials of degree $\le k$ in each variable. $Q_1$ on a square uses the bilinear functions $1, x, y, xy$ and has 4 nodes.
- $P_0$ / $Q_0$: piecewise constants, one value per cell, discontinuous.

**Continuity.** $P_k/Q_k$ with $k\ge1$ give globally continuous functions, a subspace of $H^1(\Omega)$ (conforming). $P_0$ is only in $L^2$.

**Choices for EIT.**

- The potential $u$ needs an $H^1$-conforming space: $Q_1$, $P_1$ or higher.
- The conductivity $\sigma$ can use a separate discretisation: $P_0$ (one value per cell, natural for $L^\infty$ and for [[Total Variation]] with jumps) or $Q_1/P_1$ (nodal, needed for the $H^1$ seminorm in [[Tikhonov Regularization]]).
- Structured $n\times n$ quadrilateral grids on $[-1,1]^2$ make $\sigma$ an image. Standard image processing and convolutional networks then apply directly (see [[Invariant and Equivariant Functions]]).

**Error estimates.** For smooth $u$ and $Q_k/P_k$ elements on a quasi-uniform mesh of size $h$: $\|u-u_h\|_{H^1} = \mathcal O(h^k)$ and $\|u-u_h\|_{L^2} = \mathcal O(h^{k+1})$. With discontinuous conductivities the solution has limited regularity, and the rates drop unless the mesh follows the discontinuities.

**In ModularEIT.jl:** [`FerriteDiscretization`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEITFerrite.FerriteDiscretization).

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. A. Ern, J.-L. Guermond (2004). *Theory and Practice of Finite Elements*. Springer. [doi:10.1007/978-1-4757-4355-5](https://doi.org/10.1007/978-1-4757-4355-5)
