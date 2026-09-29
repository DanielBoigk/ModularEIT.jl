---
tags: [forward-problem]
---

In the **Neumann problem** the current density on the boundary is prescribed. Given $g\in H^{-1/2}(\partial\Omega)$, find $u$ with

$$
\nabla\cdot(\gamma\nabla u) = 0 \ \text{in } \Omega, \qquad \gamma\,\partial_\nu u = g \ \text{on } \partial\Omega,
$$

where $\nu$ is the outward unit normal. Here $g$ is the current flowing *into* the body per unit boundary length or area: with $\mathbf J = -\gamma\nabla u$, it equals $-\mathbf J\cdot\nu$.

**Compatibility.** Integrating the equation over $\Omega$ and using the divergence theorem shows that a solution can only exist if

$$
\int_{\partial\Omega} g \,\mathrm ds = 0 ,
$$

that is, the injected current equals the extracted current ([[Continuity Equation|charge conservation]]).

**Non-uniqueness up to constants.** If $u$ solves the problem, so does $u + c$ for every $c\in\mathbb R$. Only potential *differences* are physical. Uniqueness is restored by [[Grounding of the Potential|grounding]]: fix $\int_{\partial\Omega}u\,\mathrm ds = 0$, or fix $u$ at one point, or work in $H^1(\Omega)/\mathbb R$.

**Weak form.** Find $u\in H^1(\Omega)/\mathbb R$ such that

$$
\int_\Omega \gamma\nabla u\cdot\nabla v\,\mathrm dx = \int_{\partial\Omega} g\, v\,\mathrm ds \qquad\forall v\in H^1(\Omega).
$$

Coercivity on $H^1(\Omega)/\mathbb R$ follows from the Poincaré–Wirtinger inequality (see [[Sobolev and Trace Spaces]] and [[Lax-Milgram Theorem]]). The derivation is in [[Weak Formulation of the Conductivity Equation]]. How the null space is handled discretely is described in [[Null Space of the Neumann Problem]].

The map $g\mapsto u|_{\partial\Omega}$ is the [[Neumann-to-Dirichlet Map]].

**In ModularEIT.jl:** [`forward_neumann`](https://danielboigk.github.io/ModularEIT.jl/dev/api/forward/#ModularEIT.forward_neumann).

## References

1. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS GSM 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
2. M. Cheney, D. Isaacson, J. C. Newell (1999). *Electrical Impedance Tomography*. SIAM Review 41(1), 85–101. [doi:10.1137/S0036144598333613](https://doi.org/10.1137/S0036144598333613)
