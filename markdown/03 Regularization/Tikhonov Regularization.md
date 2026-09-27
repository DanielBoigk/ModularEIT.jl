---
tags: [regularization, classical]
aliases: [Tikhonov, L2 regularization, H1 regularization]
---

**Tikhonov regularisation** penalises a quadratic norm of the conductivity or of its deviation from a reference $\sigma^*$:

$$
\mathcal R_{L^2}(\sigma) = \tfrac12\|\sigma-\sigma^*\|_{L^2(\Omega)}^2,
\qquad
\mathcal R_{H^1}(\sigma) = \tfrac12\|\nabla\sigma\|_{L^2(\Omega)}^2 = \tfrac12\int_\Omega |\nabla\sigma|^2\,\mathrm dx .
$$

The $L^2$ version prefers small deviations. The $H^1$ seminorm version prefers *smooth* conductivities and does not penalise constants.

**Discretisation.** With $\sigma_h = \sum_i z_i\varphi_i$ in a finite element space,

$$
\mathcal R_{L^2}(z) = \tfrac12 (z-z^*)^\top M (z-z^*), \qquad \mathcal R_{H^1}(z) = \tfrac12 z^\top K z,
$$

with the [[Mass Matrix]] $M$ and the [[Stiffness Matrix]] $K$. For $\beta\mathcal R_{H^1}$ the gradient is $\beta Kz$ and the Hessian $\beta K$. The $H^1$ version needs a continuous space ($P_1/Q_1$ or higher); for piecewise constant $\sigma$ one uses a discrete gradient (jumps across faces) instead.

**Proximal operator.** For use in [[ADMM]] (see [[Proximal Operator]]):

$$
\operatorname{prox}(y) = \arg\min_z\ \tfrac\beta2 z^\top K z + \tfrac\rho2\|z-y\|^2
\quad\Longleftrightarrow\quad (\beta K + \rho I)\,z = \rho\,y .
$$

If the proximity term is measured in the $L^2(\Omega)$ norm $\|z-y\|_M^2$, the system becomes $(\beta K+\rho M)z = \rho M y$. The matrix is symmetric positive definite, so the problem is strongly convex with a unique minimiser, and [[Conjugate Gradient Method|CG]] solves it efficiently.

**Properties.** It is simple, convex and differentiable, and is the MAP estimate under a Gaussian prior. Its drawback is that edges are blurred. For piecewise constant targets use [[Total Variation]].

In the linearised setting, the Tikhonov solution of $J\delta = r$ is $\delta = (J^\top J + \beta L^\top L)^{-1}J^\top r$. This is the same system that appears in the [[Levenberg-Marquardt Method]].

## References

1. A. N. Tikhonov (1963). *On the solution of ill-posed problems and the method of regularization*. Dokl. Akad. Nauk SSSR 151(3), 501–504 (English transl.: Soviet Math. Dokl. 4, 1035–1038). [mathnet.ru/eng/dan28329](https://www.mathnet.ru/eng/dan28329)
2. H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
3. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
