---
tags: [inverse-problem, theory]
aliases: [CGO solutions]
---

**Complex geometrical optics (CGO) solutions** are the main tool behind the uniqueness and reconstruction results for the [[Calderón Problem]].

**Reduction to Schrödinger form.** For smooth $\gamma$, the substitution $u = \gamma^{-1/2} w$ turns $\nabla\cdot(\gamma\nabla u) = 0$ into

$$
(-\Delta + q)\,w = 0, \qquad q = \frac{\Delta\sqrt\gamma}{\sqrt\gamma}.
$$

By [[Boundary Determination]], $\Lambda_\gamma$ determines the DtN map of the Schrödinger operator, so it suffices to recover $q$.

**CGO solutions.** Let $\zeta\in\mathbb C^n$ with $\zeta\cdot\zeta = 0$, for example $\zeta = \tau(\omega + i\omega^\perp)$ with orthogonal unit vectors. Then $e^{x\cdot\zeta}$ is harmonic, and one looks for solutions of the form

$$
w(x) = e^{x\cdot\zeta}\big(1 + r(x,\zeta)\big), \qquad \|r\| = \mathcal O(|\zeta|^{-1}) \text{ as } |\zeta|\to\infty .
$$

These solutions grow exponentially in one direction and oscillate in another.

**Uniqueness argument ($n\ge3$).** An integral identity (Alessandrini's identity) gives $\int_\Omega (q_1-q_2)\,w_1 w_2\,\mathrm dx = 0$ whenever the DtN maps agree. In $n\ge3$ one can choose $\zeta_1+\zeta_2 = i\xi$ for any $\xi\in\mathbb R^n$ with $|\zeta_j|\to\infty$. Then $w_1w_2\to e^{i x\cdot\xi}$, the Fourier transform of $q_1-q_2$ vanishes, and so $q_1 = q_2$.

In two dimensions there is not enough freedom to choose such $\zeta$. Nachman instead used $\bar\partial$-methods in the complex spectral parameter (see [[D-bar Method]]).

## References

1. J. Sylvester, G. Uhlmann (1987). *A Global Uniqueness Theorem for an Inverse Boundary Value Problem*. Ann. of Math. 125(1), 153–169. [doi:10.2307/1971291](https://doi.org/10.2307/1971291)
2. A. I. Nachman (1988). *Reconstructions From Boundary Measurements*. Ann. of Math. 128(3), 531–576. [doi:10.2307/1971435](https://doi.org/10.2307/1971435)
3. G. Uhlmann (2009). *Electrical impedance tomography and Calderón's problem*. Inverse Problems 25(12), 123011. [doi:10.1088/0266-5611/25/12/123011](https://doi.org/10.1088/0266-5611/25/12/123011)
