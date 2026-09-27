---
tags: [numerics, fem, forward-problem]
aliases: [L_gamma, EIT system matrix]
---

The system matrix of the discretised [[Conductivity Equation]] is the **conductivity-weighted stiffness matrix**

$$
(L_\gamma)_{ij} = \int_\Omega\gamma(x)\,\nabla\varphi_i\cdot\nabla\varphi_j\,\mathrm dx .
$$

With Neumann data $g$, the discrete forward problem is $L_\gamma\mathbf u = \mathbf g$ with the load vector $\mathbf g_i = \int_{\partial\Omega}g\,\varphi_i\,\mathrm ds$.

**Assembly.** Loop over cells and interpolate $\gamma$ at the quadrature points $x_q$. The conductivity may live on its own mesh or space, for example $P_0$, or $Q_1$ on the same grid:

$$
L_e[i,j] = \sum_q \gamma(x_q)\,\nabla\varphi_i(x_q)\cdot\nabla\varphi_j(x_q)\,w_q\,|\det J_e(x_q)| ,
$$

then scatter into the global sparse matrix (see [[Numerical Quadrature and Assembly]]).

**Properties.**

- Symmetric positive semidefinite with kernel = constants, like the [[Stiffness Matrix]].
- Spectrally equivalent to $K$: $\gamma_{\min}K\le L_\gamma\le\gamma_{\max}K$.
- **Linear in $\gamma$**: $L_\gamma = \sum_k\gamma_k L^{(k)}$ for a basis expansion $\gamma = \sum_k\gamma_k\psi_k$. The solution $\mathbf u = L_\gamma^{-1}\mathbf g$ is nonlinear in $\gamma$ nonetheless (see [[Forward Map]]).
- Its sparsity pattern does not depend on $\gamma$, so the symbolic structure and the preconditioner setup can be reused across iterations.

Every reconstruction iteration reassembles $L_\sigma$ for the current guess $\sigma$ and then solves the state and adjoint systems with it (see [[Adjoint State Method]]).

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. A. Adler, W. R. B. Lionheart (2006). *Uses and abuses of EIDORS: an extensible software base for EIT*. Physiol. Meas. 27(5), S25–S42. [doi:10.1088/0967-3334/27/5/S03](https://doi.org/10.1088/0967-3334/27/5/S03)
