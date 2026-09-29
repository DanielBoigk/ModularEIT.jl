---
tags: [numerics, fem]
aliases: [Assembly, Quadrature]
---

Finite element matrices are computed **cell by cell** and then **assembled** into a global sparse matrix.

**Reference element mapping.** Each cell $\Omega_e$ is the image of a reference cell $\hat K$ under a map $F_e$ with Jacobian $J_e$. Shape functions and gradients transform as

$$
\varphi_i = \hat\varphi_i\circ F_e^{-1},\qquad \nabla\varphi_i = J_e^{-\top}\hat\nabla\hat\varphi_i .
$$

**Quadrature.** Integrals over the reference cell are replaced by weighted sums at quadrature points $\hat x_q$ with weights $w_q$:

$$
\int_{\Omega_e} f\,\mathrm dx \approx \sum_q f(F_e(\hat x_q))\,w_q\,|\det J_e(\hat x_q)| .
$$

Gauss rules with $m$ points per direction integrate polynomials of degree $2m-1$ exactly. For $Q_1$ stiffness matrices with a constant or $Q_1$ conductivity, $2\times2$ Gauss points are standard.

**Assembly loop.**

```text
A ← 0
for each cell e:
    A_e ← local matrix from quadrature
    A[dofs(e), dofs(e)] += A_e
```

The same pattern assembles the [[Mass Matrix]], the [[Stiffness Matrix]], the [[Weighted Stiffness Matrix]], load vectors, and the gradient of the data misfit (see [[Functional Derivative of the Data Misfit]]). Boundary terms use a loop over boundary facets with facet quadrature (see [[Boundary Mass and Stiffness Matrices]]).

The sparsity pattern depends only on the mesh and the degrees of freedom. It can be computed once and the values overwritten in each iteration.

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. K. Carlsson, F. Ekre, and contributors. *Ferrite.jl* (software). [github.com/Ferrite-FEM/Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
