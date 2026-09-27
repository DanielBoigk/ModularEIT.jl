---
tags: [numerics, fem]
aliases: [Adaptive meshing, Residual estimator]
---

A finite element solution $u_h$ satisfies the discrete equations exactly. What remains is the residual of the *continuous* problem, the functional

$$
\mathcal r(v) = \ell(v) - a(u_h,v) = \int_{\partial\Omega}g\,v\,\mathrm ds - \int_\Omega\sigma\nabla u_h\cdot\nabla v\,\mathrm dx,
$$

which vanishes for $v\in V_h$ (Galerkin orthogonality) but not for general $v\in H^1$. Note that the algebraic residual $\mathbf g - L_\sigma\mathbf u_h$ is zero, up to solver tolerance, and says nothing about discretisation error.

**Residual-based estimator.** Integrating by parts cell-wise gives local indicators

$$
\eta_K^2 = h_K^2\,\|\nabla\cdot(\sigma\nabla u_h)\|^2_{L^2(K)} + \sum_{F\subset\partial K} h_F\,\big\|[\![\sigma\partial_\nu u_h]\!]\big\|^2_{L^2(F)},
$$

where the second term contains the jumps of the normal flux across interior faces. On boundary faces the jump is replaced by $g - \sigma\partial_\nu u_h$. Then $\|u-u_h\|_{a}\lesssim\big(\sum_K\eta_K^2\big)^{1/2}$ (reliability), and $\eta_K$ also bounds the local error from below (efficiency).

**Adaptive loop.** SOLVE → ESTIMATE → MARK (for example Dörfler marking: the smallest set of cells carrying a fixed fraction of the total error) → REFINE ($h$-refinement by splitting, or $p$-refinement by raising the polynomial degree).

**In EIT.** Errors concentrate near electrodes, where currents are singular at the electrode edges, and at conductivity jumps. Goal-oriented (dual-weighted residual) estimators, which weight residuals with the adjoint solution, target exactly the error in the measured boundary voltages.

## References

1. M. Ainsworth, J. T. Oden (2000). *A Posteriori Error Estimation in Finite Element Analysis*. Wiley. [doi:10.1002/9781118032824](https://doi.org/10.1002/9781118032824)
2. R. Becker, R. Rannacher (2001). *An optimal control approach to a posteriori error estimation in finite element methods*. Acta Numerica 10, 1–102. [doi:10.1017/S0962492901000010](https://doi.org/10.1017/S0962492901000010)
3. W. Dörfler (1996). *A Convergent Adaptive Algorithm for Poisson's Equation*. SIAM J. Numer. Anal. 33(3), 1106–1124. [doi:10.1137/0733054](https://doi.org/10.1137/0733054)
