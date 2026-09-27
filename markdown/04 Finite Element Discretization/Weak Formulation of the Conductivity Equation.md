---
tags: [numerics, fem, forward-problem]
aliases: [Weak form]
---

Start from the [[Neumann Problem]]

$$
\nabla\cdot(\gamma\nabla u) = 0 \ \text{in }\Omega,\qquad \gamma\,\partial_\nu u = g\ \text{on }\partial\Omega .
$$

**Step 1: test.** Multiply by a test function $v\in H^1(\Omega)$ and integrate. This is the $L^2(\Omega)$ inner product with $v$:

$$
\int_\Omega v\,\nabla\cdot(\gamma\nabla u)\,\mathrm dx = 0 .
$$

**Step 2: integrate by parts.** [[Green's Identities|Green's first identity]] with $\mathbf F = \gamma\nabla u$ gives

$$
\int_\Omega v\,\nabla\cdot\mathbf F\,\mathrm dx = \int_{\partial\Omega} v\,\mathbf F\cdot\nu\,\mathrm ds - \int_\Omega\nabla v\cdot\mathbf F\,\mathrm dx .
$$

Since $\gamma\nabla u\cdot\nu = \gamma\,\partial_\nu u$:

$$
\int_{\partial\Omega} v\,\gamma\,\partial_\nu u\,\mathrm ds - \int_\Omega\gamma\,\nabla u\cdot\nabla v\,\mathrm dx = 0 .
$$

**Step 3: insert the boundary condition.** The Neumann condition is *natural*: it enters through the boundary integral.

$$
\boxed{\ \int_\Omega\gamma\,\nabla u\cdot\nabla v\,\mathrm dx = \int_{\partial\Omega} g\,v\,\mathrm ds\qquad\forall v\in H^1(\Omega).\ }
$$

Only first derivatives appear, so $u\in H^1(\Omega)$ and $\gamma\in L^\infty(\Omega)$ suffice. Testing with $v\equiv1$ gives the compatibility condition $\int_{\partial\Omega} g = 0$.

**Dirichlet case.** For $u = f$ on $\partial\Omega$ the condition is *essential*: it is built into the trial space ($\operatorname{tr}u = f$). The test functions vanish on the boundary ($v\in H^1_0$), so the boundary integral disappears:

$$
\int_\Omega\gamma\nabla u\cdot\nabla v\,\mathrm dx = 0\qquad\forall v\in H^1_0(\Omega).
$$

**With interior sources** $h$: add $\int_\Omega h\,v$ to the right-hand side.

Existence and uniqueness follow from the [[Lax-Milgram Theorem]]. Discretisation leads to the [[Galerkin Method]].

## References

1. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS GSM 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
2. S. C. Brenner, L. R. Scott (2008). *The Mathematical Theory of Finite Element Methods*, 3rd ed. Springer. [doi:10.1007/978-0-387-75934-0](https://doi.org/10.1007/978-0-387-75934-0)
