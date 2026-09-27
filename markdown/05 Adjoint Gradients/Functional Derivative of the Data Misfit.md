---
tags: [adjoint, gradient]
aliases: [EIT gradient, Adjoint gradient]
---

Varying the [[Lagrangian Formulation|weak Lagrangian]] in the conductivity, $\sigma\to\sigma+\varepsilon\,\delta\sigma$:

$$
\partial_\sigma\mathcal L[\delta\sigma] = -\int_\Omega\delta\sigma\ \nabla u\cdot\nabla\lambda\,\mathrm dx .
$$

$\sigma$ enters the weak Lagrangian linearly and in a single term, so this is exact. No integration by parts is needed, no boundary terms arise, and no assumption on $\delta\sigma|_{\partial\Omega}$ is required.

Evaluated at the solutions $u$ of the [[State Equation]] and $\lambda$ of the [[Adjoint Equation]], this is the derivative of the reduced functional (see [[Adjoint State Method]]). Its $L^2(\Omega)$ representative is

$$
\boxed{\ \nabla_\sigma J\big(u(\sigma)\big) = -\nabla u\cdot\nabla\lambda\quad\text{a.e. in }\Omega .\ }
$$

For several current patterns, sum over the patterns: $\nabla J = -\sum_i\nabla u_i\cdot\nabla\lambda_i$. Each term is the [[Linearized EIT and the Sensitivity Kernel|sensitivity kernel]] contracted with the residual.

**Discrete version.** With $\sigma_h = \sum_a z_a\psi_a$ the derivative with respect to the coefficients is

$$
\frac{\partial J}{\partial z_a} = -\int_\Omega\psi_a\ \nabla u_h\cdot\nabla\lambda_h\,\mathrm dx ,
$$

assembled cell by cell with quadrature (see [[Numerical Quadrature and Assembly]]). For $P_0$ conductivities this is simply $-\int_{K_a}\nabla u_h\cdot\nabla\lambda_h$. The vector is a *covector*. Turning it into a function in the $\sigma$-space requires the Riesz map, for example the [[L2 Projection]] with the mass matrix (see [[Gradient Representation and the Riesz Map]]).

**Adding a regulariser.** If the objective contains $\beta\mathcal R(\sigma)$, the gradient becomes $\beta\nabla\mathcal R(\sigma) - \sum_i\nabla u_i\cdot\nabla\lambda_i$. The state and adjoint equations do not change.

## References

1. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
2. R.-E. Plessix (2006). *A review of the adjoint-state method for computing the gradient of a functional with geophysical applications*. Geophys. J. Int. 167(2), 495–503. [doi:10.1111/j.1365-246X.2006.02978.x](https://doi.org/10.1111/j.1365-246X.2006.02978.x)
3. F. J. Margotti (2015). *On Inexact Newton Methods for Inverse Problems in Banach Spaces*. PhD thesis, KIT. [doi:10.5445/IR/1000048606](https://doi.org/10.5445/IR/1000048606)
