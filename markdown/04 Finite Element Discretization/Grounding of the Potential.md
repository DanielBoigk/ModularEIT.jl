---
tags: [numerics, fem, forward-problem]
aliases: [Grounding, Grounding condition, Reference potential]
---

The solution of the [[Neumann Problem]] is unique only up to an additive constant: if $u$ solves it, so does $u+c$. Only potential *differences* are physical. A **grounding condition** is one linear functional $\ell$ with $\ell(1)\ne0$ that selects one representative:

$$
\ell(u) = 0 .
$$

**Common choices**

| grounding | continuous | discrete ($w^\top\mathbf u = 0$) |
|:--|:--|:--|
| zero mean over the domain | $\int_\Omega u\,\mathrm dx = 0$ | $w = M\mathbf 1$ |
| zero mean on the boundary | $\int_{\partial\Omega}u\,\mathrm ds = 0$ | $w = M_\Gamma\mathbf 1$ |
| zero sum of boundary nodal values | – | $w_i = 1$ for boundary DOFs, else $0$ |
| zero sum of all nodal values | – | $w = \mathbf 1$ (Euclidean minimum norm) |
| one grounded node / electrode | $u(x_0) = 0$ | $w = e_{i_0}$ |
| CEM electrode voltages | $\sum_\ell U_\ell = 0$ | see [[Complete Electrode Model]] |

The boundary sum and the boundary mean agree on a uniform boundary mesh. In general they differ by the weights of the [[Boundary Mass and Stiffness Matrices|boundary mass matrix]].

**Changing the grounding afterwards.** All solutions differ by constants, so any solution $\tilde{\mathbf u}$ can be regrounded with the (oblique) projection

$$
\mathbf u = \tilde{\mathbf u} - \mathbf 1\,\frac{w^\top\tilde{\mathbf u}}{w^\top\mathbf 1}.
$$

For a general null space with basis $V$ and $k$ grounding functionals $W$ this becomes $\mathbf u = \tilde{\mathbf u} - V(W^\top V)^{-1}W^\top\tilde{\mathbf u}$, which requires $W^\top V$ to be invertible. A solver can therefore work with whatever representative is most convenient, for example the one orthogonal to the null space (see [[Projected Conjugate Gradient]]) or the one with a pinned node (see [[Projected Cholesky Factorization]]), and ground at the end.

**Why the boundary grounding is natural for EIT.**

- The [[Neumann-to-Dirichlet Map]] is defined with $\int_{\partial\Omega}u = 0$: only boundary voltages are measured, so the reference should be fixed by boundary values, not by the (unmeasured) interior.
- The boundary misfit $\|u|_{\partial\Omega}-f\|^2$ is *not* invariant under $u\mapsto u+c$. Minimising it over the constant gives $\int_{\partial\Omega}(u+c-f) = 0$. If the measured voltages $f$ have zero boundary mean, grounding $u$ with zero boundary mean automatically picks the best-fitting constant, so the misfit only measures the physically meaningful voltage differences. With Euclidean nodal norms the same holds for the zero *sum* of boundary nodal values.
- Grounding with the interior mean, or with the full nodal sum, adds a constant offset that depends on the conductivity in the interior. That spurious offset then appears in the data misfit.

## References

1. P. Bochev, R. B. Lehoucq (2005). *On the Finite Element Solution of the Pure Neumann Problem*. SIAM Review 47(1), 50–66. [doi:10.1137/S0036144503426074](https://doi.org/10.1137/S0036144503426074)
2. E. Somersalo, M. Cheney, D. Isaacson (1992). *Existence and Uniqueness for Electrode Models for Electric Current Computed Tomography*. SIAM J. Appl. Math. 52(4), 1023–1040. [doi:10.1137/0152060](https://doi.org/10.1137/0152060)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
