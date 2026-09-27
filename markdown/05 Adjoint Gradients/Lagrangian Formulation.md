---
tags: [optimization, adjoint]
aliases: [Lagrangian]
---

For the constraint "u solves the [[Neumann Problem]] with conductivity $\sigma$, current $g$ and interior source $h$", the **Lagrangian** couples the objective with the constraint through a multiplier $\lambda$ (the adjoint state):

$$
\mathcal L(\sigma,u,\lambda) = J(u) + \langle\lambda, e(\sigma,u)\rangle .
$$

**Strong form.** $e(\sigma,u) = \nabla\cdot(\sigma\nabla u)+h$ gives $\mathcal L = J(u)+\int_\Omega\lambda\,(\nabla\cdot(\sigma\nabla u)+h)\,\mathrm dx$.

**Weak form (preferred).** Integrating by parts once and inserting $\sigma\partial_\nu u = g$ gives

$$
\mathcal L(\sigma,u,\lambda) = J(u) - \int_\Omega\sigma\,\nabla u\cdot\nabla\lambda\,\mathrm dx + \int_\Omega h\,\lambda\,\mathrm dx + \int_{\partial\Omega} g\,\lambda\,\mathrm ds .
$$

Advantages of the weak form:

1. The Neumann datum $g$ appears explicitly.
2. Every variation needs at most one integration by parts. Terms with $\partial_\nu\delta u$, which have no trace for $\delta u\in H^1$, never appear.
3. $\sigma$ appears only in one term, which is linear in $\sigma$, so the $\sigma$-variation is exact and produces no boundary terms.

**Function spaces.** Trial space $V$ for $u$ and space $V_0$ of admissible variations:

| forward problem | $V$ | $V_0$ |
|:--|:--|:--|
| Dirichlet, $u = f$ | $\{v\in H^1: v\vert_{\partial\Omega}=f\}$ | $H^1_0(\Omega)$ |
| Neumann, $\sigma\partial_\nu u = g$ | $H^1(\Omega)/\mathbb R$ | $H^1(\Omega)/\mathbb R$ |

The multiplier lives in $V_0$. For the Dirichlet problem drop the boundary integral, since $\lambda\in H^1_0$.

Setting the variations to zero gives the [[State Equation]], the [[Adjoint Equation]] and the [[Functional Derivative of the Data Misfit|functional derivative]] (see [[KKT Conditions]] and [[Adjoint State Method]]).

## References

1. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
2. A. M. Bradley (2024). *PDE-constrained optimization and the adjoint method*. Lecture notes, Stanford. [cs.stanford.edu/~ambrad/adjoint_tutorial.pdf](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf)
