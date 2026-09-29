---
tags: [adjoint]
---

Varying the [[Lagrangian Formulation|weak Lagrangian]] in the state, $u\to u+\varepsilon\,\delta u$ with $\delta u\in V_0$:

$$
\partial_u\mathcal L[\delta u] = J'(u)[\delta u] - \int_\Omega\sigma\,\nabla\delta u\cdot\nabla\lambda\,\mathrm dx = 0 \qquad\forall\delta u\in V_0 .
$$

Split the derivative of the misfit into interior and boundary Riesz representers,

$$
J'(u)[\delta u] = \int_\Omega(\partial_uJ)_\Omega\,\delta u\,\mathrm dx + \int_{\partial\Omega}(\partial_uJ)_{\partial\Omega}\,\delta u\,\mathrm ds .
$$

**Weak adjoint equation.** Find $\lambda\in V_0$ with

$$
\int_\Omega\sigma\nabla\lambda\cdot\nabla v\,\mathrm dx = \int_\Omega(\partial_uJ)_\Omega\,v\,\mathrm dx + \int_{\partial\Omega}(\partial_uJ)_{\partial\Omega}\,v\,\mathrm ds\qquad\forall v\in V_0 .
$$

It has the *same bilinear form* as the state equation, because the conductivity operator is self-adjoint. So the same matrix $L_\sigma$, factorisation or preconditioner is reused, and only the right-hand side changes.

**Strong form** (integrating by parts and using the two-stage argument as in [[State Equation]]):

$$
\nabla\cdot(\sigma\nabla\lambda) = -(\partial_uJ)_\Omega\ \ \text{in }\Omega,\qquad \sigma\,\partial_\nu\lambda = (\partial_uJ)_{\partial\Omega}\ \ \text{on }\partial\Omega .
$$

**Standard EIT misfit.** For $J(u) = \|u-f\|^2_{L^2(\partial\Omega)}$ (no interior term):

$$
\nabla\cdot(\sigma\nabla\lambda) = 0\ \text{in }\Omega,\qquad\sigma\,\partial_\nu\lambda = 2\,(u-f)\ \text{on }\partial\Omega .
$$

The adjoint is itself an EIT forward problem, driven by the voltage residual as boundary current. For a Neumann adjoint the compatibility condition requires the right-hand side to have zero mean. This holds when both $u$ and $f$ are grounded to zero boundary mean. $\lambda$ is determined up to a constant, which does not affect $\nabla\lambda$.

**Sign convention.** With the Lagrangian written as $J+\int\sigma\nabla u\cdot\nabla\lambda-\dots$ instead, $\lambda$ changes sign, and so does the formula for the gradient. The product $-\nabla u\cdot\nabla\lambda$ in [[Functional Derivative of the Data Misfit]] belongs to the convention used here.

**Dirichlet case.** $V_0 = H^1_0$, so $\lambda = 0$ on $\partial\Omega$. The measured quantity is the current, and the misfit enters through the flux instead.

**In ModularEIT.jl:** [`AdjointStateObjective`](https://danielboigk.github.io/ModularEIT.jl/dev/api/objectives/#ModularEIT.AdjointStateObjective).

## References

1. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
2. D. Lahaye, W. Mulckhuyse (2012). *Adjoint sensitivity in PDE constrained least squares problems as a multiphysics problem*. COMPEL 31(3), 895–903. [doi:10.1108/03321641211209780](https://doi.org/10.1108/03321641211209780)
3. F. J. Margotti (2015). *On Inexact Newton Methods for Inverse Problems in Banach Spaces*. PhD thesis, KIT, Sec. 5.2.2. [doi:10.5445/IR/1000048606](https://doi.org/10.5445/IR/1000048606)
