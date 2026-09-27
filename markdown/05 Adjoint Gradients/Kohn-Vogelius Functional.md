---
tags: [objective, adjoint]
aliases: [Equation error functional, Kohn–Vogelius]
---

Instead of comparing boundary voltages, the **Kohn–Vogelius functional** compares *two interior fields* driven by the same measured Cauchy pair $(f,g)$:

- $u_f$ solves the [[Dirichlet Problem]] with $u_f = f$ on $\partial\Omega$;
- $u_g$ solves the [[Neumann Problem]] with $\sigma\partial_\nu u_g = g$ on $\partial\Omega$.

If $\sigma$ is the true conductivity, $u_f = u_g$ up to a constant. Otherwise their difference measures the inconsistency everywhere in $\Omega$:

$$
J_{\mathrm{KV}}(\sigma) = \frac12\int_\Omega\sigma\,|\nabla u_f-\nabla u_g|^2\,\mathrm dx .
$$

It depends only on $\nabla(u_f-u_g)$, so the undetermined constant in $u_g$ drops out.

**Energy identity.** With the [[Dirichlet and Thomson Principles]] and the measured power $P = \int_{\partial\Omega}f\,g\,\mathrm ds$ (independent of $\sigma$):

$$
2J_{\mathrm{KV}}(\sigma) = \langle f,\Lambda_\sigma f\rangle + \langle g,\Lambda_\sigma^{-1}g\rangle - 2P .
$$

The cross term follows from testing the Neumann weak form with $v = u_f$: $\int\sigma\nabla u_f\cdot\nabla u_g = \int_{\partial\Omega} g\,f = P$.

**Gradient without adjoints.** Both energies are minima over fields constrained only by the data, so by the envelope theorem only the explicit $\sigma$-dependence contributes:

$$
\nabla_\sigma J_{\mathrm{KV}} = \tfrac12\big(|\nabla u_f|^2 - |\nabla u_g|^2\big).
$$

One iteration costs two forward solves (one Dirichlet, one Neumann) and **no adjoint solve**. At the true conductivity, the misfit corresponds to a data fit in the natural energy ($H^{1/2}$-type) norm rather than in $L^2(\partial\Omega)$.

**Relaxation and regularisation.** Minimising sequences of the unregularised functional can oscillate finer and finer. Kohn and Vogelius studied its relaxation (homogenisation), which leads to anisotropic, non-unique limits (see [[Anisotropic Conductivities]]). In practice the unrelaxed functional is used with bounds $\sigma_{\min}\le\sigma\le\sigma_{\max}$ and an explicit regulariser. Kohn and McKenney reported that early termination has a desirable smoothing effect.

## References

1. R. V. Kohn, M. Vogelius (1987). *Relaxation of a variational method for impedance computed tomography*. Comm. Pure Appl. Math. 40(6), 745–777. [doi:10.1002/cpa.3160400605](https://doi.org/10.1002/cpa.3160400605)
2. R. V. Kohn, A. McKenney (1990). *Numerical implementation of a variational method for electrical impedance tomography*. Inverse Problems 6(3), 389–414. [doi:10.1088/0266-5611/6/3/009](https://doi.org/10.1088/0266-5611/6/3/009)
3. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136, Sec. 7.2. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
4. L. Borcea (2003). *Addendum to "Electrical impedance tomography"*. Inverse Problems 19(4), 997–998. [doi:10.1088/0266-5611/19/4/501](https://doi.org/10.1088/0266-5611/19/4/501)
