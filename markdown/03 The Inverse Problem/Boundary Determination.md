---
tags: [inverse-problem, theory]
---

The first uniqueness result for the [[Calderón Problem]] concerns the boundary itself. Kohn and Vogelius (1984) showed that for smooth conductivities, $\Lambda_\gamma$ determines $\gamma|_{\partial\Omega}$ and all its normal derivatives $\partial_\nu^k\gamma|_{\partial\Omega}$.

**Idea.** Use boundary voltages that oscillate rapidly and are concentrated near a boundary point $x_0$. The corresponding solutions decay quickly away from the boundary, so $\langle\Lambda_\gamma f,f\rangle$ only sees $\gamma$ in a thin layer near $x_0$. In the limit this gives $\gamma(x_0)$; higher-order asymptotics give the normal derivatives. In the language of microlocal analysis, $\Lambda_\gamma$ is a pseudodifferential operator of order one whose full symbol determines the Taylor series of $\gamma$ at the boundary (Sylvester–Uhlmann 1988).

**Consequences.**

- Real-analytic conductivities are uniquely determined, because their Taylor series at the boundary determines them. Kohn and Vogelius (1985) extended this to piecewise analytic conductivities.
- The same localisation explains why high-frequency [[Current Patterns]] only carry information about the region near the boundary.

## References

1. R. V. Kohn, M. Vogelius (1984). *Determining conductivity by boundary measurements*. Comm. Pure Appl. Math. 37(3), 289–298. [doi:10.1002/cpa.3160370302](https://doi.org/10.1002/cpa.3160370302)
2. R. V. Kohn, M. Vogelius (1985). *Determining conductivity by boundary measurements II. Interior results*. Comm. Pure Appl. Math. 38(5), 643–667. [doi:10.1002/cpa.3160380513](https://doi.org/10.1002/cpa.3160380513)
3. J. Sylvester, G. Uhlmann (1988). *Inverse boundary value problems at the boundary—continuous dependence*. Comm. Pure Appl. Math. 41(2), 197–219. [doi:10.1002/cpa.3160410205](https://doi.org/10.1002/cpa.3160410205)
