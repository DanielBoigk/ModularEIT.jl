---
tags: [optimization]
aliases: [Limited-memory BFGS]
---

**BFGS** is a quasi-Newton method. It builds an approximation $H_k\approx(\nabla^2\Phi)^{-1}$ of the inverse Hessian from gradient differences and steps along $p_k = -H_k\nabla\Phi(\sigma_k)$. With $s_k = \sigma_{k+1}-\sigma_k$, $y_k = \nabla\Phi_{k+1}-\nabla\Phi_k$ and $\rho_k = 1/(y_k^\top s_k)$:

$$
H_{k+1} = (I-\rho_ks_ky_k^\top)\,H_k\,(I-\rho_ky_ks_k^\top)+\rho_ks_ks_k^\top .
$$

The update keeps $H$ positive definite as long as $y_k^\top s_k>0$, which a Wolfe [[Line Search]] guarantees.

**Limited memory (L-BFGS).** Dense $H_k$ is impossible for $10^4$–$10^6$ unknowns. L-BFGS stores only the last $m$ pairs $(s_k,y_k)$, typically $m = 5$–$20$, and applies $H_k$ with the *two-loop recursion* in $\mathcal O(mn)$ operations. The initial matrix $H_0 = \gamma_kI$ with $\gamma_k = \frac{s^\top y}{y^\top y}$ sets the scale. A problem-adapted $H_0$, such as the inverse [[Mass Matrix]], makes the method mesh-independent (see [[Gradient Representation and the Riesz Map]]).

**For EIT.** It needs only objective values and gradients, one state and one adjoint solve per pattern and evaluation (see [[Adjoint State Method]]). Curvature is learned along the way. It usually converges much faster than steepest descent, with superlinear local convergence. For bound constraints use [[L-BFGS-B]].

**In ModularEIT.jl:** [`LBFGS`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.LBFGS).

## References

1. D. C. Liu, J. Nocedal (1989). *On the limited memory BFGS method for large scale optimization*. Math. Program. 45, 503–528. [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
2. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed., Ch. 6–7. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
