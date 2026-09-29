---
tags: [numerics, fem, adaptivity]
aliases: [ZZ estimator, Recovery-based error estimator, Gradient recovery]
---

The **Zienkiewicz–Zhu (ZZ) estimator** compares the discrete gradient with a *recovered*, smoother gradient. For linear or bilinear elements, $\nabla u_h$ is discontinuous across cell boundaries, while the exact gradient is (piecewise) smooth. Averaging $\nabla u_h$ into a continuous field $G^*$ gives a better approximation of $\nabla u$ than $\nabla u_h$ itself (superconvergence). Their difference then estimates the error:

$$
\eta_K^2 = \int_K \big|G^* - \nabla u_h\big|^2\,\mathrm dx,\qquad \|\nabla(u-u_h)\|_{L^2}\approx\Big(\sum_K\eta_K^2\Big)^{1/2}.
$$

The recovery can be a nodal average, a local least-squares fit on the patch of cells around each node (superconvergent patch recovery), or the [[L2 Projection]] of $\nabla u_h$ onto continuous piecewise linear vector fields.

**Discontinuous conductivity: recover the current density.** Across a jump of $\sigma$, the exact gradient itself jumps, so recovering $\nabla u$ smears the jump and flags the interface forever. The quantity that is continuous across interfaces is the normal **current density** $J = \sigma\nabla u$. The estimator for the conductivity equation therefore recovers $J^*$ from $J_h = \sigma\nabla u_h$:

$$
\eta_K^2 = \sum_s\int_K\big|J_s^* - \sigma\nabla u_{h,s}\big|^2\,\mathrm dx ,
$$

summed over the current patterns $s$. The tangential component of $J$ still jumps at interfaces, so interfaces keep being refined, but only as long as they contribute to the error.

**Properties.** The estimator is cheap: one mass-matrix solve per pattern and vector component, independent of the equation. It is asymptotically exact for smooth solutions on sufficiently regular meshes. It is not guaranteed to be reliable on coarse meshes or at singularities, where residual estimators (see [[A Posteriori Error Estimation and Adaptive Meshing]]) have rigorous bounds. For an exact discrete solution, for example a linear $u$, it vanishes.

**In ModularEIT.jl:** [`flux_recovery_indicator`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEIT.flux_recovery_indicator).

## References

1. O. C. Zienkiewicz, J. Z. Zhu (1987). *A simple error estimator and adaptive procedure for practical engineering analysis*. Int. J. Numer. Methods Eng. 24(2), 337–357. [doi:10.1002/nme.1620240206](https://doi.org/10.1002/nme.1620240206)
2. O. C. Zienkiewicz, J. Z. Zhu (1992). *The superconvergent patch recovery and a posteriori error estimates. Part 1: The recovery technique*. Int. J. Numer. Methods Eng. 33(7), 1331–1364. [doi:10.1002/nme.1620330702](https://doi.org/10.1002/nme.1620330702)
3. M. Ainsworth, J. T. Oden (2000). *A Posteriori Error Estimation in Finite Element Analysis*. Wiley. [doi:10.1002/9781118032824](https://doi.org/10.1002/9781118032824)
