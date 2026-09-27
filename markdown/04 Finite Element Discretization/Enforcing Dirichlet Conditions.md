---
tags: [numerics, fem]
---

For the [[Dirichlet Problem]], the boundary values are prescribed at the boundary degrees of freedom $B$. Let $I$ denote the interior DOFs. Partition the system $L\mathbf u = \mathbf b$:

$$
\begin{pmatrix} L_{II} & L_{IB}\\ L_{BI} & L_{BB}\end{pmatrix}\begin{pmatrix}\mathbf u_I\\ \mathbf u_B\end{pmatrix} = \begin{pmatrix}\mathbf b_I\\ \mathbf b_B\end{pmatrix},\qquad \mathbf u_B = \mathbf f .
$$

**Elimination (symmetric).** Solve $L_{II}\mathbf u_I = \mathbf b_I - L_{IB}\mathbf f$. The reduced matrix $L_{II}$ is symmetric positive definite, so CG applies.

**Row/column replacement (same size).** Keep the full system but, for each $i\in B$, zero out row $i$ and column $i$, set the diagonal to $1$, set $b_i = f_i$, and move the eliminated column contributions $-L_{IB}\mathbf f$ to the right-hand side. This keeps the matrix size and symmetry and is equivalent to elimination. Zeroing only the rows is simpler, but it destroys symmetry.

**Penalty / Nitsche.** Add $\frac{\kappa}{h}\int_{\partial\Omega}(u-f)v$ with a large penalty. This is approximate unless the symmetric Nitsche terms are included.

**Measured quantity.** In the Dirichlet setting one measures the boundary current $g = \gamma\partial_\nu u$. The consistent discrete way to obtain it is the *residual* of the unconstrained boundary rows, $\mathbf r_B = (L\mathbf u)_B$. By the weak form, $(r_B)_i \approx \int_{\partial\Omega} g\,\varphi_i\,\mathrm ds$, so the nodal current density follows from $M_\Gamma\mathbf g = \mathbf r_B$ (see [[Boundary Mass and Stiffness Matrices]]). This variational flux recovery is more accurate than differentiating $u_h$ numerically.

## References

1. A. Ern, J.-L. Guermond (2004). *Theory and Practice of Finite Elements*. Springer. [doi:10.1007/978-1-4757-4355-5](https://doi.org/10.1007/978-1-4757-4355-5)
2. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
