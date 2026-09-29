---
tags: [adjoint, gradient]
aliases: [Voltage-driven adjoint, Adjoint for DtN data]
---

In voltage-driven EIT, voltages $f$ are prescribed on the boundary (or on the electrodes) and the currents are measured. The data are samples of the [[Dirichlet-to-Neumann Map]] $\Lambda_\sigma$. The least-squares misfit is

$$
J(\sigma) = \frac12\sum_s\big\Vert\Lambda_\sigma f_s - g_s\big\Vert^2 .
$$

**Derivative of the DtN map.** Let $u_f$, $u_v$ solve the [[Dirichlet Problem]] with data $f$ and $v$. Testing the equation for $u_f$ with $u_v$ gives $\langle v,\Lambda_\sigma f\rangle = \int_\Omega\sigma\nabla u_f\cdot\nabla u_v\,\mathrm dx$. Differentiate in $\sigma$: the derivatives of $u_f$ and $u_v$ vanish on the boundary, and both fields satisfy the equation, so their contributions drop out. What remains is

$$
\langle v,\Lambda'_\sigma[\delta\sigma]\,f\rangle = \int_\Omega\delta\sigma\,\nabla u_f\cdot\nabla u_v\,\mathrm dx .
$$

For the [[Neumann-to-Dirichlet Map]] the sign is opposite: $\langle h,\mathcal R'_\sigma[\delta\sigma]\,g\rangle = -\int\delta\sigma\,\nabla u_g\cdot\nabla u_h$. More conductivity means more current for a given voltage, and less voltage for a given current.

**Gradient.** Take $v = r_s = \Lambda_\sigma f_s - g_s$. Then

$$
\nabla J = \sum_s \nabla u_{f_s}\cdot\nabla\lambda_s,\qquad \lambda_s = \text{Dirichlet solution with boundary data } r_s .
$$

The adjoint is another Dirichlet problem with the residual as boundary data. Its operator is the same as the state's, so one factorisation serves both.

**Discrete version.** With the free and Dirichlet degrees of freedom $F$, $B$ (see [[Discrete Electrode Models]]), the discrete DtN map is the Schur complement $S(\sigma) = A_{BB}-A_{BF}A_{FF}^{-1}A_{FB}$. The identity $\mathbf v^\top S\,\mathbf f = \tilde{\mathbf x}_{\mathbf v}^\top A\,\mathbf x_{\mathbf f}$ holds for the discrete harmonic extensions $\mathbf x_{\mathbf f}$, $\tilde{\mathbf x}_{\mathbf v}$. The same stationarity argument then gives

$$
\frac{\partial}{\partial\sigma_a}\big(\mathbf v^\top S(\sigma)\,\mathbf f\big) = \tilde{\mathbf x}_{\mathbf v}^\top\,\frac{\partial A}{\partial\sigma_a}\,\mathbf x_{\mathbf f},
$$

a contraction with the [[Conductivity Tensor]]. If the currents are compared in another representation $C^{-1}E^\top(A\mathbf x)_B$, the adjoint boundary data become $C^{-1}\mathbf r$. For a weighted misfit with $W = U^\top U$, they become $C^{-1}U^\top\mathbf r$.

**Jacobian.** Replacing $\mathbf r$ by the unit vectors of the measurements gives one Dirichlet solve per measurement. These are independent of the pattern $s$, so a single block solve gives the Jacobian for all patterns.

**In ModularEIT.jl:** [`AdjointStateObjective`](https://danielboigk.github.io/ModularEIT.jl/dev/api/objectives/#ModularEIT.AdjointStateObjective).

## References

1. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
