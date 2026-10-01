---
tags: [numerics, fem, boundary]
aliases: [M_Gamma, K_Gamma]
---

Boundary data (voltages $f$, currents $g$) are functions on $\partial\Omega$. With the traces of the basis functions restricted to the boundary DOFs, define

$$
(M_\Gamma)_{ij} = \int_{\partial\Omega}\varphi_i\varphi_j\,\mathrm ds,\qquad
(K_\Gamma)_{ij} = \int_{\partial\Omega}\nabla_\Gamma\varphi_i\cdot\nabla_\Gamma\varphi_j\,\mathrm ds,
$$

where $\nabla_\Gamma$ is the tangential (surface) gradient. They are assembled like the volume matrices, but by looping over boundary facets with facet quadrature (see [[Numerical Quadrature and Assembly]]).

- $M_\Gamma$ is the $L^2(\partial\Omega)$ Gram matrix: $\|f_h\|^2_{L^2(\partial\Omega)} = \mathbf f^\top M_\Gamma\mathbf f$.
- $K_\Gamma$ is the discrete Laplace–Beltrami operator on the boundary curve or surface. It is positive semidefinite, with kernel = constants on each connected boundary component.

**Why it matters.** The Euclidean product $\mathbf a^\top\mathbf b$ of nodal values is *not* the $L^2(\partial\Omega)$ inner product. It over-weights regions with fine boundary meshes. Consistent choices are:

- [[Data Fidelity Terms]]: $\tfrac12(\mathbf u-\mathbf f)^\top M_\Gamma(\mathbf u-\mathbf f)$. The adjoint right-hand side is then $M_\Gamma(\mathbf u-\mathbf f)$.
- Load vector of a current density $g$: $\mathbf g = M_\Gamma\,\mathbf g_{\text{nodal}}$ (interpolate, then multiply).
- Mean zero: $\mathbf 1^\top M_\Gamma\mathbf f = 0$.
- SVD of boundary operators in the correct geometry: transform with $M_\Gamma^{1/2}$ (see [[Truncated SVD Regularization]]).

The pair $(K_\Gamma, M_\Gamma)$ also defines the [[Discrete Fractional Sobolev Norms]].

**In ModularEIT.jl:** [`assemble_boundary_mass!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEITFerrite.assemble_boundary_mass!), [`FEMatrices`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.FEMatrices).

## References

1. M. Arioli, D. Loghin (2009). *Discrete Interpolation Norms with Applications*. SIAM J. Numer. Anal. 47(4), 2924–2951. [doi:10.1137/080729360](https://doi.org/10.1137/080729360)
2. A. Ern, J.-L. Guermond (2004). *Theory and Practice of Finite Elements*. Springer. [doi:10.1007/978-1-4757-4355-5](https://doi.org/10.1007/978-1-4757-4355-5)
