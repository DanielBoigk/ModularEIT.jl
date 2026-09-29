---
tags: [numerics, fem]
aliases: [Galerkin discretization, Finite element method, FEM]
---

A **Galerkin method** approximates a variational problem

$$
\text{find } u\in V:\quad a(u,v) = \ell(v)\quad\forall v\in V
$$

by replacing the infinite-dimensional space $V$ with a finite-dimensional subspace $V_h = \operatorname{span}\{\varphi_1,\dots,\varphi_n\}$. Writing $u_h = \sum_j u_j\varphi_j$ and testing with each $\varphi_i$ gives the linear system

$$
A\,\mathbf u = \mathbf b,\qquad A_{ij} = a(\varphi_j,\varphi_i),\quad b_i = \ell(\varphi_i).
$$

For the [[Conductivity Equation]], $a(u,v) = \int_\Omega\gamma\nabla u\cdot\nabla v$, so $A$ is the [[Weighted Stiffness Matrix]] $L_\gamma$. With Neumann data, $\ell(v) = \int_{\partial\Omega}g\,v$.

**Choice of basis.**

- *Finite elements*: piecewise polynomials on a mesh, giving sparse matrices and complex geometries (see [[Lagrange Finite Elements]]);
- *spectral / Fourier / Chebyshev bases*: dense but very accurate on simple domains;
- *wavelets*.

**Céa's lemma.** If $a$ is bounded (constant $C$) and coercive (constant $\alpha$) on $V$, the Galerkin solution is quasi-optimal:

$$
\|u-u_h\|_V \le \frac{C}{\alpha}\,\inf_{v_h\in V_h}\|u-v_h\|_V .
$$

The approximation error of the space therefore controls the discretisation error. For EIT, $C/\alpha = \gamma_{\max}/\gamma_{\min}$ (see [[Lax-Milgram Theorem]]).

In an EIT reconstruction the Galerkin solve is the inner step of each iteration (see [[Iterative Reconstruction Loop]]).

**In ModularEIT.jl:** [`FerriteDiscretization`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.FerriteDiscretization).

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. A. Ern, J.-L. Guermond (2004). *Theory and Practice of Finite Elements*. Springer. [doi:10.1007/978-1-4757-4355-5](https://doi.org/10.1007/978-1-4757-4355-5)
