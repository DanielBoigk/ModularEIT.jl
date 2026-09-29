---
tags: [analysis]
aliases: [Integration by parts, Divergence theorem]
---

Let $\Omega$ be a bounded Lipschitz domain with outward unit normal $\nu$.

**Divergence theorem.** For a vector field $\mathbf F$,
$$ \int_\Omega\nabla\cdot\mathbf F\,\mathrm dx = \int_{\partial\Omega}\mathbf F\cdot\nu\,\mathrm ds .$$

**Green's first identity** (integration by parts). Apply the divergence theorem to $v\,\kappa\nabla u$:
$$ \int_\Omega v\,\nabla\cdot(\kappa\nabla u)\,\mathrm dx = -\int_\Omega\kappa\,\nabla u\cdot\nabla v\,\mathrm dx + \int_{\partial\Omega}v\,\kappa\,\partial_\nu u\,\mathrm ds .$$

**Green's second identity** (symmetric form):
$$ \int_\Omega\big(v\,\nabla\cdot(\kappa\nabla u) - u\,\nabla\cdot(\kappa\nabla v)\big)\mathrm dx = \int_{\partial\Omega}\kappa\,(v\,\partial_\nu u - u\,\partial_\nu v)\,\mathrm ds .$$

**Uses in EIT.**

- Deriving the [[Weak Formulation of the Conductivity Equation]].
- Showing that the [[Dirichlet-to-Neumann Map]] is symmetric (second identity with two solutions; see [[Properties of the Boundary Operators]]).
- Deriving the [[Linearized EIT and the Sensitivity Kernel|linearisation identity]] and the [[Adjoint Equation]].

**General adjoint identity.** For a linear differential operator $P$ with formal adjoint $P^*$:
$\int_\Omega\lambda\,(Pv) = \int_\Omega(P^*\lambda)\,v + B(\lambda,v)|_{\partial\Omega}$. The boundary form $B$ determines the boundary conditions of adjoint problems (see [[Rules of the Calculus of Variations]]).

## References

1. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS GSM 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
