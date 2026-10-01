---
tags: [numerics, fem, adaptivity, adjoint]
aliases: [Dual-weighted residual, DWR]
---

Energy-norm estimators control the error everywhere in $\Omega$. EIT, however, only uses a few numbers: the electrode voltages. **Goal-oriented** estimation controls the error of a quantity of interest $\mathcal Q(u)$ directly.

**Dual-weighted residual.** Let $a(u,v) = \ell(v)$ be the forward problem, $u_h$ its Galerkin solution, and $\mathcal Q$ linear. Let $z$ solve the dual (adjoint) problem $a(v,z) = \mathcal Q(v)$ for all $v$. Then

$$
\mathcal Q(u) - \mathcal Q(u_h) = a(u-u_h,z) = \ell(z) - a(u_h,z) = \mathcal r(z - z_h)
$$

for any $z_h$ in the finite element space, by Galerkin orthogonality. $\mathcal r$ is the residual functional. Splitting $\mathcal r(z-z_h)$ into cell contributions gives indicators $\eta_K = \rho_K(u_h)\,\omega_K(z)$: the local residual of the forward solution weighted by the local approximation error of the dual solution. Cells are refined only where both are large, i.e. where errors are made *and* matter for $\mathcal Q$.

**For EIT the duals are already there.** The measured voltage $V_m = Q_m\mathbf u$ is a linear functional of the state, and its dual problem is the adjoint problem of the [[Adjoint State Method]] with a unit misfit on measurement $m$. These are the adjoint fields that build the Jacobian rows (see [[Conductivity Tensor]]). One set of dual solves serves all current patterns. For the data misfit itself, the dual is the adjoint state $\lambda$ of the gradient computation. The DWR estimator then controls the error in the objective, and the same framework yields error estimators for the reconstructed conductivity (Becker and Vexler).

**Practical points.** The weight $z - z_h$ must be approximated, for example by a higher-order recovery of $z_h$ or by solving the dual problem on a finer space. The estimator is not guaranteed to be an upper bound, but it is usually much sharper for $\mathcal Q$ than energy-norm estimators.

**A simple product indicator.** By Cauchy–Schwarz, $|a(u-u_h,z-z_h)|\le\sum_K\|u-u_h\|_{a,K}\,\|z-z_h\|_{a,K}$. Estimating both local energy errors with the [[Zienkiewicz-Zhu Estimator]] gives the indicator $\eta_K = \eta_K(u)\,\eta_K(z)$. Summed over patterns and measurements, it needs no residual evaluation. It refines only where the primal error is large *and* the measurements are sensitive, and it vanishes when either solution is resolved exactly.

**In ModularEIT.jl:** [`goal_oriented_indicator`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEITFerrite.goal_oriented_indicator).

## References

1. R. Becker, R. Rannacher (2001). *An optimal control approach to a posteriori error estimation in finite element methods*. Acta Numerica 10, 1–102. [doi:10.1017/S0962492901000010](https://doi.org/10.1017/S0962492901000010)
2. R. Becker, B. Vexler (2004). *A posteriori error estimation for finite element discretization of parameter identification problems*. Numer. Math. 96(3), 435–459. [doi:10.1007/s00211-003-0482-9](https://doi.org/10.1007/s00211-003-0482-9)
3. M. Ainsworth, J. T. Oden (2000). *A Posteriori Error Estimation in Finite Element Analysis*. Wiley. [doi:10.1002/9781118032824](https://doi.org/10.1002/9781118032824)
