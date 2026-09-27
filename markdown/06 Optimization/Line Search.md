---
tags: [optimization]
aliases: [Step size, Brent's method, Wolfe conditions]
---

Given a descent direction $p$ (with $\nabla\Phi^\top p<0$), a **line search** picks the step $\tau>0$ in $\sigma_{k+1} = \sigma_k+\tau p$.

**Inexact line search (Wolfe conditions).** Accept $\tau$ if

$$
\Phi(\sigma+\tau p)\le\Phi(\sigma)+c_1\tau\,\nabla\Phi^\top p\quad\text{(sufficient decrease / Armijo)},
$$
$$
\nabla\Phi(\sigma+\tau p)^\top p\ge c_2\,\nabla\Phi^\top p\quad\text{(curvature)},
$$

with $0<c_1<c_2<1$ (typically $c_1 = 10^{-4}$, $c_2 = 0.9$). Backtracking ($\tau\leftarrow\tau/2$ until Armijo holds) is the simplest variant. Hager–Zhang and Moré–Thuente are robust implementations. The Wolfe conditions guarantee that quasi-Newton updates stay positive definite (see [[L-BFGS]]).

**Exact 1D minimisation (Brent's method).** It minimises $\phi(\tau) = \Phi(\sigma+\tau p)$ on a bracket by combining golden-section search with parabolic interpolation, without derivatives. Every evaluation is a full forward solve for all patterns, so it is expensive. It is still useful after a [[Gauss-Newton Method|Gauss–Newton]] direction, where the natural step length is unclear because of nonlinearity.

**Projected search.** With bounds $\sigma_{\min}\le\sigma\le\sigma_{\max}$, evaluate $\Phi(P(\sigma+\tau p))$ with the projection $P$ onto the box (see [[Box Constraints on Conductivity]]).

## References

1. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed., Ch. 3. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
2. R. P. Brent (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall (Dover reprint 2002). [ISBN 978-0-486-41998-5](https://search.worldcat.org/search?q=bn:9780486419985)
3. W. W. Hager, H. Zhang (2006). *Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent*. ACM Trans. Math. Softw. 32(1), 113–137. [doi:10.1145/1132973.1132979](https://doi.org/10.1145/1132973.1132979)
