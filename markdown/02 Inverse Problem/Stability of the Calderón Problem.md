---
tags: [inverse-problem, theory]
aliases: [Stability, Logarithmic stability]
---

**Question.** If $\Lambda_{\gamma_1}$ and $\Lambda_{\gamma_2}$ are close, are $\gamma_1$ and $\gamma_2$ close?

We look for an estimate of the form

$$
\|\gamma_1-\gamma_2\|_{L^\infty(\Omega)} \le \omega\big(\|\Lambda_{\gamma_1}-\Lambda_{\gamma_2}\|_{H^{1/2}\to H^{-1/2}}\big)
$$

with a modulus of continuity $\omega(t)\to0$ as $t\to0$.

**Logarithmic stability.** Alessandrini (1988) proved, for $n\ge3$ and conductivities satisfying a priori bounds (for example in $H^s$, $s>n/2+2$), that

$$
\omega(t) = C\,|\log t|^{-\delta}\qquad (t<1)
$$

for some $\delta\in(0,1)$. Mandache (2001) showed that this rate cannot be improved in general. Halving the reconstruction error requires the data error to shrink to about its $2^{1/\delta}$-th power. This is why EIT is called **severely ill-posed** (see [[Well-Posedness]]).

**Better stability under stronger priors.** If $\gamma$ is known to lie in a finite-dimensional set, for example piecewise constant on a known partition into finitely many cells, then the estimate is **Lipschitz**: $\|\gamma_1-\gamma_2\|\le C\|\Lambda_{\gamma_1}-\Lambda_{\gamma_2}\|$ (Alessandrini–Vessella 2005). The constant $C$, however, grows exponentially with the number of unknowns. This is the theoretical case for strong prior information, both hand-crafted and learned (see [[Variational Regularization]] and [[Learned Regularization]]).

**Practical meaning.** Features deep inside the domain, or small ones, produce boundary signals that are exponentially small. With finite measurement precision, no algorithm can resolve them without prior assumptions.

## References

1. G. Alessandrini (1988). *Stable determination of conductivity by boundary measurements*. Applicable Analysis 27(1–3), 153–172. [doi:10.1080/00036818808839730](https://doi.org/10.1080/00036818808839730)
2. N. Mandache (2001). *Exponential instability in an inverse problem for the Schrödinger equation*. Inverse Problems 17(5), 1435–1444. [doi:10.1088/0266-5611/17/5/313](https://doi.org/10.1088/0266-5611/17/5/313)
3. G. Alessandrini, S. Vessella (2005). *Lipschitz stability for the inverse conductivity problem*. Adv. Appl. Math. 35(2), 207–241. [doi:10.1016/j.aam.2004.12.002](https://doi.org/10.1016/j.aam.2004.12.002)
4. J. L. Mueller, S. Siltanen (2012). *Linear and Nonlinear Inverse Problems with Practical Applications*. SIAM. [doi:10.1137/1.9781611972344](https://doi.org/10.1137/1.9781611972344)
