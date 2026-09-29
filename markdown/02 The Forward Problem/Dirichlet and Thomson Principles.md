---
tags: [forward-problem, variational]
aliases: [Dirichlet principle, Thomson principle]
---

Both forward problems have energy characterisations.

**Dirichlet principle.** The solution $u_f$ of the [[Dirichlet Problem]] minimises the dissipated power among all potentials with the prescribed boundary voltage:

$$
\langle f,\Lambda_\gamma f\rangle = \min_{v\in H^1,\ v|_{\partial\Omega}=f}\ \int_\Omega \gamma|\nabla v|^2\,\mathrm dx = \int_\Omega\gamma|\nabla u_f|^2\,\mathrm dx .
$$

**Thomson principle.** The current density $\mathbf J = -\gamma\nabla u_g$ of the [[Neumann Problem]] minimises the dissipated power among all divergence-free current fields with the prescribed boundary flux:

$$
\langle g,\Lambda_\gamma^{-1} g\rangle = \min_{\substack{\nabla\cdot\mathbf k=0\\ -\mathbf k\cdot\nu = g}}\ \int_\Omega \gamma^{-1}|\mathbf k|^2\,\mathrm dx = \int_\Omega\gamma|\nabla u_g|^2\,\mathrm dx .
$$

**Consequences.**

- *Monotonicity*: since the integrands increase in $\gamma$ (Dirichlet) or in $\gamma^{-1}$ (Thomson), larger conductivity means larger $\Lambda_\gamma$ and smaller $\Lambda^{-1}_\gamma$ (see [[Properties of the Boundary Operators]]).
- *Derivatives*: by the envelope theorem, only the explicit $\gamma$-dependence survives differentiation. The derivative of $\langle f,\Lambda_\gamma f\rangle$ in direction $\delta\gamma$ is $\int\delta\gamma|\nabla u_f|^2$, and that of $\langle g,\Lambda^{-1}_\gamma g\rangle$ is $-\int\delta\gamma|\nabla u_g|^2$. These formulas give the gradient of the [[Kohn-Vogelius Functional]] without any adjoint solve.

## References

1. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
2. R. V. Kohn, M. Vogelius (1987). *Relaxation of a variational method for impedance computed tomography*. Comm. Pure Appl. Math. 40(6), 745–777. [doi:10.1002/cpa.3160400605](https://doi.org/10.1002/cpa.3160400605)
