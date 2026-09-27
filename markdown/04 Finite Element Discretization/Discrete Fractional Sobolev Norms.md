---
tags: [numerics, fem, boundary]
aliases: [H^1/2 norm, H^-1/2 norm, Discrete interpolation norms]
---

The natural spaces for boundary voltages and currents are $H^{1/2}(\partial\Omega)$ and $H^{-1/2}(\partial\Omega)$ (see [[Sobolev and Trace Spaces]]). They can be computed discretely by **spectral interpolation** between $L^2$ and $H^1$ on the boundary.

Solve the generalised eigenproblem with the [[Boundary Mass and Stiffness Matrices]]

$$
K_\Gamma\,\Phi = M_\Gamma\,\Phi\,\operatorname{diag}(\lambda_k),\qquad \Phi^\top M_\Gamma\Phi = I,\quad \lambda_k\ge0 .
$$

Clip tiny negative eigenvalues caused by round-off to zero. The columns of $\Phi$ are the discrete Laplace–Beltrami eigenfunctions (on a circle: Fourier modes, with $\lambda_k = k^2$). For $s\in[-1,1]$ define

$$
A(s) = M_\Gamma\,\Phi\,\operatorname{diag}\big((1+\lambda_k)^s\big)\,\Phi^\top M_\Gamma ,
\qquad \|\mathbf f\|_{H^s}^2 = \mathbf f^\top A(s)\,\mathbf f .
$$

- $s=0$: $A(0) = M_\Gamma$, the $L^2$ norm.
- $s=1$: $A(1) = M_\Gamma + K_\Gamma$, the $H^1$ norm.
- $s=\pm\tfrac12$: the discrete $H^{\pm1/2}$ norms. On a circle, $\|f\|^2_{H^s}\simeq\sum_k(1+k^2)^s|\hat f_k|^2$.

**Duality.** Since $\Phi^\top M_\Gamma\Phi = I$, one gets $A(s)^{-1} = \Phi\operatorname{diag}((1+\lambda_k)^{-s})\Phi^\top$ and hence

$$
A(-s) = M_\Gamma\,A(s)^{-1}\,M_\Gamma .
$$

So the $H^{-1/2}$ matrix is the $L^2$-dual of the $H^{1/2}$ matrix, as in the continuous setting.

**Uses in EIT.** Weighting the voltage misfit in $H^{1/2}$ or the current misfit in $H^{-1/2}$ matches the mapping properties of the [[Neumann-to-Dirichlet Map]] and changes how high-frequency boundary data are weighted relative to low-frequency data (see [[Data Fidelity Terms]]).

## References

1. M. Arioli, D. Loghin (2009). *Discrete Interpolation Norms with Applications*. SIAM J. Numer. Anal. 47(4), 2924–2951. [doi:10.1137/080729360](https://doi.org/10.1137/080729360)
2. W. McLean (2000). *Strongly Elliptic Systems and Boundary Integral Equations*. Cambridge University Press. [ISBN 978-0-521-66375-5](https://search.worldcat.org/search?q=bn:9780521663755)
