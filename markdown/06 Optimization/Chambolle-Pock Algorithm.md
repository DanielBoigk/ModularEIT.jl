---
tags: [optimization, splitting]
aliases: [Primal-dual hybrid gradient, PDHG]
---

The **Chambolle–Pock** (primal–dual hybrid gradient) algorithm solves saddle-point problems

$$
\min_x\max_p\ \langle Kx,p\rangle+G(x)-F^*(p),
$$

which is the primal–dual form of $\min_x F(Kx)+G(x)$. It needs only products with $K$ and $K^\top$ and the proximal operators of $G$ and $F^*$.

**Iteration** (step sizes $\tau,s>0$ with $\tau s\|K\|^2<1$, $\theta = 1$):

$$
\begin{aligned}
p^{k+1} &= \operatorname{prox}_{sF^*}\big(p^k+s\,K\bar x^k\big),\\
x^{k+1} &= \operatorname{prox}_{\tau G}\big(x^k-\tau K^\top p^{k+1}\big),\\
\bar x^{k+1} &= x^{k+1}+\theta\,(x^{k+1}-x^k).
\end{aligned}
$$

**TV denoising (ROF prox).** For $\min_z\beta\operatorname{TV}(z)+\frac\rho2\|z-y\|^2$ take $K = \nabla$ (discrete gradient), $F = \beta\|\cdot\|_{2,1}$ and $G = \frac\rho2\|\cdot-y\|^2$. Then:

- $\operatorname{prox}_{sF^*}$ is the pointwise projection of $p$ onto the ball $|p|\le\beta$;
- $K^\top = -\operatorname{div}$;
- $\operatorname{prox}_{\tau G}(v) = \dfrac{v+\tau\rho\,y}{1+\tau\rho}$.

For the standard forward-difference gradient on a unit grid, $\|\nabla\|^2\le8$ in 2D.

This computes the [[Total Variation]] prox exactly, without smoothing. It can be used as the regulariser step in [[ADMM]].

## References

1. A. Chambolle, T. Pock (2011). *A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging*. J. Math. Imaging Vis. 40, 120–145. [doi:10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1)
2. A. Chambolle, V. Caselles, D. Cremers, M. Novaga, T. Pock (2010). *An Introduction to Total Variation for Image Analysis*. In: Theoretical Foundations and Numerical Methods for Sparse Recovery, de Gruyter, 263–340. [doi:10.1515/9783110226157.263](https://doi.org/10.1515/9783110226157.263)
