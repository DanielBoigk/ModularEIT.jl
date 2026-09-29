---
tags: [adjoint, numerics]
aliases: [DTO vs OTD]
---

**Optimise-then-discretise (OTD).** Derive the optimality system (state, adjoint, gradient) in function spaces, as in [[Lagrangian Formulation]], then discretise each equation. The discrete gradient approximates the continuous gradient $-\nabla u\cdot\nabla\lambda$, but it is **not necessarily the exact gradient of the discrete objective**.

**Discretise-then-optimise (DTO).** Discretise the objective and PDE first: $\min_{\mathbf z}J(\mathbf u)$ s.t. $L_{\mathbf z}\mathbf u = \mathbf g$. Then differentiate the finite-dimensional problem exactly. The discrete adjoint is

$$
L_{\mathbf z}^\top\boldsymbol\lambda = \nabla_{\mathbf u}J,\qquad \frac{\partial J}{\partial z_a} = -\boldsymbol\lambda^\top\frac{\partial L_{\mathbf z}}{\partial z_a}\mathbf u = -\int_\Omega\psi_a\nabla u_h\cdot\nabla\lambda_h\,\mathrm dx
$$

(the last equality holds if the quadrature is exact for the integrand). This is what [[Automatic Differentiation vs Adjoint Methods|AD]] produces.

**When do they coincide?** For the conductivity equation with a Galerkin discretisation, the two agree whenever the gradient is assembled with the same quadrature as $L_{\mathbf z}$ and represented in the dual (coefficient) sense. Differences appear when

- the gradient is interpolated or [[L2 Projection|projected]] onto a space in which $-\nabla u\cdot\nabla\lambda$ does not lie;
- different quadrature rules are used for assembly and gradient;
- solvers are stopped early (inexact states and adjoints);
- the misfit is measured in a norm other than the one used for discretisation.

Both gradients come from the same assembled quantity. The DTO gradient is the dual vector $\big(-\int\psi_a\nabla u_h\cdot\nabla\lambda_h\big)_a$, and the $L^2$-projected OTD gradient is $M_\sigma^{-1}$ times it (see [[Conductivity Tensor]]).

DTO gradients make line searches and quasi-Newton methods behave consistently, because they are exact derivatives of the function being minimised. OTD gradients are mesh-independent approximations of the true gradient (see [[Gradient Representation and the Riesz Map]]).

**In ModularEIT.jl:** [`CoefficientGradient`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.CoefficientGradient), [`L2Gradient`](https://danielboigk.github.io/ModularEIT.jl/dev/api/discretization/#ModularEIT.L2Gradient).

## References

1. M. D. Gunzburger (2002). *Perspectives in Flow Control and Optimization*. SIAM. [doi:10.1137/1.9780898718720](https://doi.org/10.1137/1.9780898718720)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
