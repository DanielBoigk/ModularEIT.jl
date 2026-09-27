---
tags: [forward-problem, boundary-operator]
aliases: [NtD map, Current-to-voltage map, Λ_γ^-1]
---

Real EIT devices inject *currents* and measure *voltages*. The matching operator is the **Neumann-to-Dirichlet (NtD) map**

$$
\mathcal R_\gamma = \Lambda_\gamma^{-1} : H^{-1/2}_\diamond(\partial\Omega)\to H^{1/2}_\diamond(\partial\Omega), \qquad g\mapsto u_g\big|_{\partial\Omega},
$$

where $u_g$ solves the [[Neumann Problem]] with current $g$ and the grounding $\int_{\partial\Omega} u_g\,\mathrm ds = 0$. The subscript $\diamond$ denotes zero-mean functions:

- the domain is restricted to zero-mean currents because of the compatibility condition $\int g = 0$;
- the range is made unique by grounding, which removes the additive constant.

On these spaces the NtD map is the inverse of the [[Dirichlet-to-Neumann Map]].

Properties (see [[Properties of the Boundary Operators]]):

- linear, self-adjoint, positive definite on $H^{-1/2}_\diamond$;
- compact as an operator on $L^2_\diamond(\partial\Omega)$: it *smooths*. For $\gamma\equiv1$ on the unit disc, $\mathcal R_1 e^{ik\theta} = |k|^{-1}e^{ik\theta}$ for $k\ne0$, so high-frequency current patterns produce small voltages (see [[Current Patterns]]);
- weak form: $\langle g, \mathcal R_\gamma h\rangle = \int_\Omega \gamma\nabla u_g\cdot\nabla u_h\,\mathrm dx$.

With finitely many current patterns $G = [g_1,\dots,g_N]$ and measured voltages $F=[f_1,\dots,f_N]$, a discrete NtD matrix is estimated as in [[Discrete Boundary Operator]].

## References

1. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
2. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
