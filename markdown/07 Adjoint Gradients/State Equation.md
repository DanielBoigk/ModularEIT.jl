---
tags: [adjoint]
---

Varying the [[Lagrangian Formulation|weak Lagrangian]]

$$
\mathcal L(\sigma,u,\lambda) = J(u) - \int_\Omega\sigma\nabla u\cdot\nabla\lambda + \int_\Omega h\lambda + \int_{\partial\Omega}g\lambda
$$

in the multiplier, $\lambda\to\lambda+\varepsilon\,\delta\lambda$, and letting $\varepsilon\to0$:

$$
\partial_\lambda\mathcal L[\delta\lambda] = -\int_\Omega\sigma\nabla u\cdot\nabla\delta\lambda\,\mathrm dx + \int_\Omega h\,\delta\lambda\,\mathrm dx + \int_{\partial\Omega}g\,\delta\lambda\,\mathrm ds = 0\qquad\forall\delta\lambda\in V_0 .
$$

This is exactly the [[Weak Formulation of the Conductivity Equation|weak forward problem]]. The **state equation** is the forward problem.

**Recovering the strong form (two-stage argument).**

1. Test with $\delta\lambda\in C_c^\infty(\Omega)$, which vanishes near the boundary, and integrate by parts. This gives $\nabla\cdot(\sigma\nabla u)+h = 0$ in $\Omega$.
2. Subtract the interior identity and allow $\delta\lambda$ with arbitrary trace. This gives the natural boundary condition $\sigma\partial_\nu u = g$ on $\partial\Omega$ (Neumann case). In the Dirichlet case the condition $u = f$ is built into the trial space.

Testing with $\delta\lambda\equiv1$ gives the compatibility condition $\int_\Omega h+\int_{\partial\Omega}g = 0$.

Solving the state equations for all [[Current Patterns]] already gives the objective value $J(u(\sigma))$. The gradient additionally needs the [[Adjoint Equation]].

**In ModularEIT.jl:** [`forward_neumann`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.forward_neumann), [`AdjointStateObjective`](https://danielboigk.github.io/ModularEIT.jl/dev/api/objectives/#ModularEIT.AdjointStateObjective).

## References

1. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
2. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS GSM 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
