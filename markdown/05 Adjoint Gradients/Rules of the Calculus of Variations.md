---
tags: [analysis, adjoint]
aliases: [Variational calculus, Gateaux derivative]
---

The manipulations used to derive the [[Adjoint State Method]].

**Definitions.**

- A *functional* $J[u]$ maps a function $u$ to a scalar.
- An *admissible variation* $\delta u$ keeps $u+\varepsilon\delta u$ admissible for small $\varepsilon$. It satisfies the homogeneous version of all *essential* boundary conditions, for example $\delta u = 0$ where $u$ is prescribed.
- The *first variation* (Gâteaux derivative) is
  $$ J'[u](\delta u) = \frac{\mathrm d}{\mathrm d\varepsilon}J[u+\varepsilon\,\delta u]\Big|_{\varepsilon=0} .$$

**Rules.**

- *Linearity:* $\delta(aF+bG) = a\,\delta F+b\,\delta G$.
- *Product and chain rule:* $\delta(FG) = (\delta F)G+F(\delta G)$, $\delta\,\Phi(u) = \Phi'(u)\,\delta u$. Only terms of first order in $\varepsilon$ are kept.
- *Commutation:* on a *fixed* domain, variation commutes with derivatives and integrals: $\delta\nabla u = \nabla\delta u$ and $\delta\int_\Omega F = \int_\Omega\delta F$. On a moving domain, a Leibniz or Reynolds transport term appears (shape derivatives).
- *Fundamental lemma:* if $\int_\Omega F\,\eta = 0$ for all $\eta\in C_c^\infty(\Omega)$, then $F = 0$ a.e. Derivatives must first be moved off $\eta$ by [[Green's Identities|integration by parts]].
- *Boundary terms decide boundary conditions:* at an essential condition, $\delta u = 0$, so the term drops and gives no condition on $\lambda$. At a natural (free) boundary, $\delta u$ is arbitrary, so its coefficient must vanish, which gives a boundary condition on $\lambda$.
- *Independence of variations:* for $\mathcal L(u,\lambda)$, a joint variation gives $\delta\mathcal L = \partial_u\mathcal L\,\delta u+\partial_\lambda\mathcal L\,\delta\lambda+\mathcal O(\varepsilon^2)$. So $\partial_u\mathcal L = 0$ and $\partial_\lambda\mathcal L = 0$ are separate conditions.

**Pitfalls.** Second variations are only needed to classify stationary points, not for gradients. $\lambda$ is a free test function only until the adjoint equation fixes it.

## References

1. I. M. Gelfand, S. V. Fomin (1963). *Calculus of Variations*. Prentice-Hall (Dover reprint 2000). [ISBN 978-0-486-41448-5](https://search.worldcat.org/search?q=bn:9780486414485)
2. M. Hinze, R. Pinnau, M. Ulbrich, S. Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [doi:10.1007/978-1-4020-8839-1](https://doi.org/10.1007/978-1-4020-8839-1)
