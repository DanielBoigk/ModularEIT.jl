---
tags: [adjoint, gradient, numerics]
aliases: [Riesz map, Sobolev gradient]
---

The derivative $\hat J'(\sigma)$ is a linear functional on perturbations $\delta\sigma$. A **gradient** is its representative with respect to an inner product $(\cdot,\cdot)_H$:

$$
(\nabla_H\hat J,\ \delta\sigma)_H = \hat J'(\sigma)[\delta\sigma]\qquad\forall\delta\sigma .
$$

The steepest-descent direction depends on the chosen metric. This is not cosmetic: it changes the iterates.

**$L^2$ gradient.** $\nabla_{L^2}\hat J = -\nabla u\cdot\nabla\lambda$ (see [[Functional Derivative of the Data Misfit]]). It is rough, and concentrated near the electrodes.

**Discrete gradients.** With coefficients $\mathbf z$ and derivative vector $\mathbf b = \partial\hat J/\partial\mathbf z$:

| metric on coefficients | gradient |
|:--|:--|
| Euclidean $\mathbf z^\top\mathbf w$ | $\mathbf b$ |
| $L^2(\Omega)$: $\mathbf z^\top M\mathbf w$ | $M^{-1}\mathbf b$ (the [[L2 Projection]]) |
| $H^1(\Omega)$: $\mathbf z^\top(M+\alpha K)\mathbf w$ | $(M+\alpha K)^{-1}\mathbf b$ |

The Euclidean gradient depends on the mesh: small cells get small entries. The $L^2$ gradient converges under mesh refinement.

**Sobolev gradients.** The $H^1$ representative is a smoothed $L^2$ gradient: it solves $(I-\alpha\Delta)\nabla_{H^1}\hat J = \nabla_{L^2}\hat J$. It acts as a preconditioner and an [[Implicit Regularization|implicit regulariser]], suppressing high-frequency updates. Quasi-Newton methods such as [[L-BFGS]] should be initialised with the metric in which the problem is posed, for example $H_0 = M^{-1}$.

## References

1. J. W. Neuberger (2010). *Sobolev Gradients and Differential Equations*, 2nd ed. Springer LNM 1670. [doi:10.1007/978-3-642-04041-2](https://doi.org/10.1007/978-3-642-04041-2)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
3. T. Schwedes, D. A. Ham, S. W. Funke, M. D. Piggott (2017). *Mesh Dependence in PDE-Constrained Optimisation*. Springer Briefs. [doi:10.1007/978-3-319-59483-5](https://doi.org/10.1007/978-3-319-59483-5)
