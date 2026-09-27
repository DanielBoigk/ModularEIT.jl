---
tags: [numerics, boundary-operator, data]
aliases: [Estimating the NtD matrix]
---

In practice one never observes the full [[Neumann-to-Dirichlet Map]], only $N$ pairs $(g_i,f_i)$. Collect them as columns, $G = [g_1\cdots g_N]$ and $F = [f_1\cdots f_N]$, with zero-mean columns (currents satisfy $\int g = 0$, voltages are grounded).

**Least-squares estimate.** The matrix $R$ with $RG\approx F$ of minimal norm is

$$
\hat R = F\,G^{+},
$$

where $G^+$ is the Moore–Penrose pseudoinverse. If the columns of $G$ are orthonormal, $G^+ = G^\top$. The estimate is only determined on $\operatorname{span}(G)$. Outside the span it acts as zero.

**Symmetrisation.** The true operator is symmetric positive semidefinite (see [[Properties of the Boundary Operators]]). Noise breaks both properties. The closest symmetric matrix is $\tfrac12(\hat R+\hat R^\top)$. Clipping negative eigenvalues then gives the closest positive semidefinite matrix in the Frobenius norm.

**Uses.**

- Generating *new* data pairs, for example optimal, orthogonal ones, through an SVD (see [[Truncated SVD Regularization]]).
- Operator-level [[Noise Models for EIT Data|noise models]].
- Operator-level data fidelity: $\|\hat R_{\text{meas}} - R_\sigma\|$.

The geometry matters: in $L^2(\partial\Omega)$ one should work with $M_\Gamma^{1/2}\hat R M_\Gamma^{1/2}$ rather than $\hat R$ (see [[Boundary Mass and Stiffness Matrices]]).

## References

1. N. J. Higham (1988). *Computing a nearest symmetric positive semidefinite matrix*. Linear Algebra Appl. 103, 103–118. [doi:10.1016/0024-3795(88)90223-6](https://doi.org/10.1016/0024-3795(88)90223-6)
2. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
