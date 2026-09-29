---
tags: [forward-problem, boundary-operator]
aliases: [DtN map, Voltage-to-current map, Λ_γ]
---

For a conductivity $\gamma$, the **Dirichlet-to-Neumann (DtN) map** sends a boundary voltage to the boundary current it produces:

$$
\Lambda_\gamma : H^{1/2}(\partial\Omega)\to H^{-1/2}(\partial\Omega), \qquad
f \mapsto \gamma\,\partial_\nu u_f\big|_{\partial\Omega},
$$

where $u_f$ solves the [[Dirichlet Problem]] with boundary value $f$.

It is the complete idealised measurement of the continuum model: knowing $\Lambda_\gamma$ means knowing the current response to *every* possible voltage pattern. Its weak definition avoids normal derivatives altogether:

$$
\langle \Lambda_\gamma f, h\rangle = \int_\Omega \gamma\,\nabla u_f\cdot\nabla u_h\,\mathrm dx ,
$$

where $u_h$ is any $H^1$ extension of $h$ (for example the solution with boundary value $h$).

Key facts, proved in [[Properties of the Boundary Operators]]:

- $\Lambda_\gamma$ is linear, bounded, self-adjoint and positive semidefinite;
- its kernel is the constants, so $\Lambda_\gamma 1 = 0$;
- its range has zero mean, and it is inverted on mean-zero functions by the [[Neumann-to-Dirichlet Map]];
- the dependence $\gamma\mapsto\Lambda_\gamma$ is **nonlinear** (see [[Forward Map]]).

**Example.** For $\gamma\equiv 1$ on the unit disc, $\Lambda_1 e^{ik\theta} = |k|\,e^{ik\theta}$. The DtN map is a first-order operator: it amplifies high spatial frequencies on the boundary.

Recovering $\gamma$ from $\Lambda_\gamma$ is the [[Calderón Problem]].

**In ModularEIT.jl:** [`forward_dirichlet`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.forward_dirichlet).

## References

1. A. P. Calderón (1980/2006). *On an inverse boundary value problem*. Reprinted in Comput. Appl. Math. 25(2–3), 133–138. [doi:10.1590/S0101-82052006000200002](https://doi.org/10.1590/S0101-82052006000200002)
2. G. Uhlmann (2009). *Electrical impedance tomography and Calderón's problem*. Inverse Problems 25(12), 123011. [doi:10.1088/0266-5611/25/12/123011](https://doi.org/10.1088/0266-5611/25/12/123011)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
