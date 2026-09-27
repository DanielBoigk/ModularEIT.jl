---
tags: [adjoint, numerics, verification]
aliases: [Taylor test, Gradient check]
---

Adjoint gradients are easy to get subtly wrong (signs, factors of 2, boundary terms, projections). The standard check is the **Taylor test**.

Pick a point $\sigma$ and a random direction $\delta\sigma$, ideally smooth and in the correct space. For a decreasing sequence $h_k = h_0 2^{-k}$ compute

$$
E_0(h) = \big|\hat J(\sigma+h\,\delta\sigma)-\hat J(\sigma)\big|,\qquad
E_1(h) = \big|\hat J(\sigma+h\,\delta\sigma)-\hat J(\sigma)-h\,\langle\nabla\hat J(\sigma),\delta\sigma\rangle\big| .
$$

- $E_0(h) = \mathcal O(h)$: convergence rate 1.
- $E_1(h) = \mathcal O(h^2)$ **if and only if** the gradient is correct: convergence rate 2, observed as $\log_2(E_1(h_k)/E_1(h_{k+1}))\approx 2$.

A wrong gradient gives rate 1 for $E_1$. Too small $h$ gives round-off plateaus, and inexact solves give a noise floor, so linear solves must be tight during the test.

Further checks:

- *Central differences* in a few coordinates: $\frac{\hat J(\sigma+he_a)-\hat J(\sigma-he_a)}{2h}\approx\partial_a\hat J$.
- *Dot-product test* for linearised operators: $\langle J\,x,y\rangle = \langle x,J^\top y\rangle$ to machine precision.
- *Symmetry of the boundary operator*: the computed NtD matrix should be symmetric up to solver tolerance (see [[Properties of the Boundary Operators]]).

## References

1. P. E. Farrell, D. A. Ham, S. W. Funke, M. E. Rognes (2013). *Automated Derivation of the Adjoint of High-Level Transient Finite Element Programs*. SIAM J. Sci. Comput. 35(4), C369–C393. [doi:10.1137/120873558](https://doi.org/10.1137/120873558)
2. A. Griewank, A. Walther (2008). *Evaluating Derivatives*, 2nd ed. SIAM. [doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
