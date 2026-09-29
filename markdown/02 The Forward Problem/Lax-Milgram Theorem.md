---
tags: [analysis, fem]
---

**Theorem (Lax–Milgram).** Let $V$ be a Hilbert space, $a:V\times V\to\mathbb R$ a bilinear form that is

- *bounded*: $|a(u,v)|\le C\|u\|_V\|v\|_V$, and
- *coercive*: $a(u,u)\ge\alpha\|u\|_V^2$ with $\alpha>0$,

and $\ell\in V^*$. Then there is exactly one $u\in V$ with $a(u,v)=\ell(v)$ for all $v\in V$, and $\|u\|_V\le\|\ell\|_{V^*}/\alpha$.

**Application to EIT.** Take $a(u,v) = \int_\Omega\gamma\nabla u\cdot\nabla v$ with $0<\gamma_{\min}\le\gamma\le\gamma_{\max}$.

- *Boundedness:* $|a(u,v)|\le\gamma_{\max}\|\nabla u\|_{L^2}\|\nabla v\|_{L^2}$.
- *Coercivity:* $a(u,u)\ge\gamma_{\min}\|\nabla u\|^2_{L^2}$. This controls the full $H^1$ norm on
  - $V = H^1_0(\Omega)$ by the Poincaré inequality ([[Dirichlet Problem]]), and
  - $V = H^1(\Omega)/\mathbb R$ by the Poincaré–Wirtinger inequality ([[Neumann Problem]]); see [[Sobolev and Trace Spaces]].
- *Right-hand side:* $\ell(v) = \int_{\partial\Omega}gv$ is bounded on $H^1$ by the trace theorem if $g\in H^{-1/2}(\partial\Omega)$. It is well defined on the quotient space iff $\int g = 0$.

So the forward problems are well-posed, with the stability estimate $\|\nabla u\|_{L^2}\le C\|g\|_{H^{-1/2}}/\gamma_{\min}$. The same argument applies to the adjoint equation (see [[Adjoint Equation]]). The lower bound $\gamma_{\min}>0$ is essential, which is one reason for [[Box Constraints on Conductivity]].

## References

1. P. D. Lax, A. N. Milgram (1954). *Parabolic equations*. In: Contributions to the Theory of Partial Differential Equations, Ann. of Math. Stud. 33, 167–190. [doi:10.1515/9781400882182-010](https://doi.org/10.1515/9781400882182-010)
2. L. C. Evans (2010). *Partial Differential Equations*, 2nd ed. AMS GSM 19. [doi:10.1090/gsm/019](https://doi.org/10.1090/gsm/019)
