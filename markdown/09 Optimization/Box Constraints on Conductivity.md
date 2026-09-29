---
tags: [optimization, physics]
aliases: [Conductivity bounds, Admissible set]
---

Physical conductivities are positive and bounded:

$$
\Sigma = \{\sigma\in L^\infty(\Omega):\ \sigma_{\min}\le\sigma(x)\le\sigma_{\max}\ \text{a.e.}\},\qquad 0<\sigma_{\min}\le\sigma_{\max}<\infty .
$$

**Why the bounds matter.**

- *Lower bound:* it makes the forward problem uniformly elliptic and coercive ([[Lax-Milgram Theorem]]). Perfect insulators ($\sigma = 0$) would disconnect the domain.
- *Upper bound:* it keeps the problem uniformly elliptic, gives continuity of the bilinear form, and excludes perfect conductors.
- *Regularisation:* bounds rule out oscillating minimising sequences and restore compactness in combination with a regulariser (see [[Kohn-Vogelius Functional]] and [[Implicit Regularization]]).
- *Prior knowledge:* typical biological tissue lies roughly between $10^{-2}$ and $2$ S/m (bone and fat low, blood and cerebrospinal fluid high). The best metallic conductor, silver, is about $6.3\times10^7$ S/m. After normalising to a reference, simulations often use $\sigma\in[\sigma_{\min},1]$ with a small $\sigma_{\min}>0$.

**Enforcing them.**

- Bound-constrained optimisers: [[L-BFGS-B]], or projected gradient steps $\sigma\leftarrow P_\Sigma(\sigma-\tau\nabla\Phi)$ with $P_\Sigma$ = clipping.
- As the regulariser $\iota_\Sigma$ in [[ADMM]], whose prox is clipping.
- Reparametrisation: $\sigma = \sigma_{\min}+e^{s}$, or a sigmoid, turns the problem into an unconstrained one. This changes the geometry of the problem and the gradient ($\partial\sigma/\partial s$ chain factor).

**In ModularEIT.jl:** [`minimize`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.minimize).

## References

1. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
2. C. Gabriel, S. Gabriel, E. Corthout (1996). *The dielectric properties of biological tissues: I. Literature survey*. Phys. Med. Biol. 41(11), 2231–2249. [doi:10.1088/0031-9155/41/11/001](https://doi.org/10.1088/0031-9155/41/11/001)
3. R. H. Byrd, P. Lu, J. Nocedal, C. Zhu (1995). *A Limited Memory Algorithm for Bound Constrained Optimization*. SIAM J. Sci. Comput. 16(5), 1190–1208. [doi:10.1137/0916069](https://doi.org/10.1137/0916069)
