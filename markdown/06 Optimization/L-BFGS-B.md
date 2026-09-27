---
tags: [optimization]
---

**L-BFGS-B** extends [[L-BFGS]] to **bound constraints** $\ell\le\sigma\le u$. In EIT these are $\sigma_{\min}\le\sigma\le\sigma_{\max}$ (see [[Box Constraints on Conductivity]]). Each iteration:

1. **Generalised Cauchy point.** Follow the projected steepest-descent path $P(\sigma-t\nabla\Phi)$, where $P$ clips to the box, and find the first local minimiser of the quadratic L-BFGS model along it. This identifies the *active set*: variables that sit at a bound.
2. **Subspace minimisation.** Minimise the quadratic model over the free variables with the active ones fixed. Then project back into the box, or truncate the step.
3. **Line search** along the resulting direction, satisfying the Wolfe conditions.

The compact limited-memory representation keeps the cost per iteration at $\mathcal O(mn)$.

**Why it matters in EIT.** Bounds keep the conductivity physical and the forward problem coercive. Simply clipping after an unconstrained step breaks the quasi-Newton curvature information. L-BFGS-B handles the bounds consistently.

## References

1. R. H. Byrd, P. Lu, J. Nocedal, C. Zhu (1995). *A Limited Memory Algorithm for Bound Constrained Optimization*. SIAM J. Sci. Comput. 16(5), 1190–1208. [doi:10.1137/0916069](https://doi.org/10.1137/0916069)
2. C. Zhu, R. H. Byrd, P. Lu, J. Nocedal (1997). *Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization*. ACM Trans. Math. Softw. 23(4), 550–560. [doi:10.1145/279232.279236](https://doi.org/10.1145/279232.279236)
