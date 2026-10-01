---
tags: [numerics, fem]
aliases: [L2-projection]
---

The **$L^2$ projection** of a function $w$ onto a finite element space $V_h=\operatorname{span}\{\varphi_i\}$ is the best approximation in $L^2(\Omega)$:

$$
P_hw = \arg\min_{v_h\in V_h}\|w-v_h\|_{L^2}
\quad\Longleftrightarrow\quad
M\,\mathbf z = \mathbf b,\qquad b_i = \int_\Omega w\,\varphi_i\,\mathrm dx ,
$$

with the [[Mass Matrix]] $M$ of $V_h$.

**Example: the adjoint gradient.** The $L^2$ gradient of the EIT data misfit is the pointwise product $-\nabla u\cdot\nabla\lambda$ (see [[Functional Derivative of the Data Misfit]]). For $u,\lambda$ in $P_p$ (or $Q_p$) it is a piecewise polynomial of higher degree ($2(p-1)$ for simplicial $P_p$), and it is discontinuous across element faces. So it generally does not lie in the conductivity space. Projecting it:

- onto piecewise constants $P_0$: exact cell averages, and $M$ is diagonal;
- onto a discontinuous space of sufficient degree: exact;
- onto continuous $P_1/Q_1$: a genuine approximation, which smooths the gradient slightly.

**Relation to the discrete gradient.** The vector $\mathbf b$ with $b_i = -\int_\Omega\nabla u\cdot\nabla\lambda\,\varphi_i$ is exactly the derivative of the discretised objective with respect to the coefficients $z_i$ of $\sigma$. The projection $M^{-1}\mathbf b$ is its $L^2$ Riesz representative. Which one to use depends on the metric of the optimiser (see [[Gradient Representation and the Riesz Map]]).

**In ModularEIT.jl:** [`l2_project`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.l2_project), [`transfer_conductivity`](https://danielboigk.github.io/ModularEIT.jl/dev/api/adaptivity/#ModularEITFerrite.transfer_conductivity).

## References

1. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
