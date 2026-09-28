---
tags: [numerics, fem, adaptivity]
aliases: [Residual indicator, Kelly estimator]
---

The residual-based estimator of [[A Posteriori Error Estimation and Adaptive Meshing]], written out for the discretised [[Conductivity Equation]] with the boundary conditions of the [[Electrode Models]]. For a discrete state $u_h$ and a current pattern with prescribed boundary current density $g$:

$$
\eta_K^2 = h_K^2\,\big\|\nabla\cdot(\sigma\nabla u_h)\big\|_{L^2(K)}^2
 + \tfrac12\sum_{F\subset\partial K\setminus\partial\Omega} h_F\,\big\|[\![\sigma\,\partial_\nu u_h]\!]\big\|_{L^2(F)}^2
 + \sum_{F\subset\partial K\cap\partial\Omega} h_F\,\big\|g-\sigma\,\partial_\nu u_h\big\|_{L^2(F)}^2 .
$$

With several current patterns, the indicators are summed, because one mesh serves all of them.

**Element residual.** $\nabla\cdot(\sigma\nabla u_h) = \nabla\sigma\cdot\nabla u_h + \sigma\,\Delta u_h$ inside a cell. It vanishes for linear elements with a piecewise constant conductivity, and for continuous $\sigma$ or higher-order elements it does not.

**Jumps of the current density.** The exact normal current density $\sigma\partial_\nu u$ is continuous across every facet, also where $\sigma$ jumps. It is therefore the *current density*, not the gradient, whose jump is measured, and a correct kink of $u$ at a conductivity interface is not flagged. On meshes with [[Hanging Nodes]], each fine piece of a coarse facet is a facet of its own. There the coarse cell's current density is evaluated at the fine facet's quadrature points.

**Boundary residuals of the electrode models.** The prescribed density $g$ comes from the electrode model:

| model | $g$ on an electrode $e_\ell$ | $g$ in the gaps |
|:--|:--|:--|
| continuum | the applied current density | — |
| [[Gap Model\|gap]] | $I_\ell/\vert e_\ell\vert$ | $0$ |
| [[Complete Electrode Model\|CEM]] | $(U_\ell-u_h)/z_\ell$ | $0$ |
| [[Point Electrode Model\|point]] | point source (not estimated) | $0$ |

On Dirichlet boundaries, as in voltage-driven problems or the [[Shunt Model]] electrodes, there is no boundary residual. The CEM residual includes the Robin condition, so it detects the steep boundary layers under electrodes with a small contact impedance.

**Reliability and efficiency.** $\|\sigma^{1/2}\nabla(u-u_h)\|_{L^2}\le C\,\big(\sum_K\eta_K^2\big)^{1/2}$, and each $\eta_K$ bounds the error on a patch around $K$ from below, up to data oscillation. With strongly discontinuous $\sigma$, the constants depend on the contrast unless the terms are weighted with the local conductivities (Bernardi and Verfürth; Petzoldt).

**Goal-oriented use.** Products of the indicators of the state and of the measurement duals give a residual-based [[Goal-Oriented Error Estimation|goal-oriented indicator]] for the electrode voltages. This is the residual counterpart of the product of [[Zienkiewicz-Zhu Estimator|recovery estimates]].

## References

1. I. Babuška, W. C. Rheinboldt (1978). *Error Estimates for Adaptive Finite Element Computations*. SIAM J. Numer. Anal. 15(4), 736–754. [doi:10.1137/0715049](https://doi.org/10.1137/0715049)
2. C. Bernardi, R. Verfürth (2000). *Adaptive finite element methods for elliptic equations with non-smooth coefficients*. Numer. Math. 85(4), 579–608. [doi:10.1007/PL00005393](https://doi.org/10.1007/PL00005393)
3. M. Petzoldt (2002). *A Posteriori Error Estimators for Elliptic Equations with Discontinuous Coefficients*. Adv. Comput. Math. 16(1), 47–75. [doi:10.1023/A:1014221125034](https://doi.org/10.1023/A:1014221125034)
4. M. Ainsworth, J. T. Oden (2000). *A Posteriori Error Estimation in Finite Element Analysis*. Wiley. [doi:10.1002/9781118032824](https://doi.org/10.1002/9781118032824)
