---
tags: [numerics, linear-algebra]
aliases: [Projected CG, CG on singular systems, CG on a quotient space]
---

Let $A\in\mathbb R^{n\times n}$ be symmetric positive **semi**definite with null space $V = \ker A$ (orthonormal basis $V\in\mathbb R^{n\times k}$), and positive definite on $V^\perp = \operatorname{range}A$. Equivalently, $A$ is an SPD operator on the quotient space $\mathbb R^n/V$. For the pure Neumann [[Weighted Stiffness Matrix]], $V = \operatorname{span}\{\mathbf 1\}$. Let

$$
\Pi = I - VV^\top
$$

be the orthogonal projector onto $V^\perp$.

**Does plain CG work?** In exact arithmetic, yes, as long as $\mathbf b\in\operatorname{range}A$ and $\mathbf x_0\in V^\perp$ (e.g. $\mathbf x_0 = 0$). All residuals and search directions then stay in $V^\perp$, where $A$ is SPD, and CG converges to the solution orthogonal to $V$ (the minimum-norm solution). The convergence rate depends on the *effective* condition number $\lambda_{\max}/\lambda_{\min}^{+}$ over the nonzero eigenvalues. Nothing in the CG recurrences is special to the nonsingular case. In practice there are three failure modes:

1. **Inconsistent right-hand side.** If $\mathbf b\notin\operatorname{range}A$, for example because the discrete load vector of a boundary current does not sum exactly to zero, then the component $V^\top\mathbf b$ can never be reduced. The residual stagnates at $\|V^\top\mathbf b\|$ and the iterate drifts along $V$ without bound. The tolerance is never reached.
2. **Round-off.** Even for consistent data, rounding errors introduce small null-space components into the residuals and directions, which accumulate over many iterations.
3. **Preconditioners.** $M^{-1}$ need not map $V^\perp$ into itself (Jacobi does not, AMG and incomplete factorisations generally do not). The preconditioned directions then acquire null-space components.

**Projected CG** removes all three by applying $\Pi$ where the iteration could leave $V^\perp$:

$$
\begin{aligned}
&\mathbf r_0 = \Pi(\mathbf b - A\mathbf x_0),\qquad \mathbf z_0 = \Pi M^{-1}\mathbf r_0,\qquad \mathbf p_0 = \mathbf z_0,\\
&\alpha_k = \frac{\mathbf r_k^\top\mathbf z_k}{\mathbf p_k^\top A\mathbf p_k},\quad \mathbf x_{k+1} = \mathbf x_k+\alpha_k\mathbf p_k,\quad \mathbf r_{k+1} = \mathbf r_k-\alpha_kA\mathbf p_k,\\
&\mathbf z_{k+1} = \Pi M^{-1}\mathbf r_{k+1},\quad \beta_k = \frac{\mathbf r_{k+1}^\top\mathbf z_{k+1}}{\mathbf r_k^\top\mathbf z_k},\quad \mathbf p_{k+1} = \mathbf z_{k+1}+\beta_k\mathbf p_k .
\end{aligned}
$$

- With inconsistent data it solves the nearest consistent problem $A\mathbf x = \Pi\mathbf b$, the least-squares solution. The removed part $\|(I-\Pi)\mathbf b\|/\|\mathbf b\|$ should be reported as a compatibility defect.
- The effective preconditioner is $\Pi M^{-1}\Pi$. It must be symmetric positive definite on $V^\perp$, which holds for a symmetric V-cycle or Jacobi.
- For $V = \operatorname{span}\{\mathbf 1\}$, applying $\Pi$ just subtracts the mean: $O(n)$ per application.
- Periodically recomputing the true residual $\Pi(\mathbf b - A\mathbf x_k)$ limits drift of the updated residual.

**Choosing the representative.** Projected CG returns the solution orthogonal to $V$. A different [[Grounding of the Potential|grounding]], such as a zero sum over the boundary nodes, is obtained afterwards by an oblique projection along $V$. This is cheaper and better conditioned than building the constraint into the matrix, which is what pinning a node or adding $\varepsilon I$ do (see [[Null Space of the Neumann Problem]]).

For many right-hand sides at once see [[Block Conjugate Gradient]].

## References

1. E. F. Kaasschieter (1988). *Preconditioned conjugate gradients for solving singular systems*. J. Comput. Appl. Math. 24(1–2), 265–275. [doi:10.1016/0377-0427(88)90358-5](https://doi.org/10.1016/0377-0427(88)90358-5)
2. P. Bochev, R. B. Lehoucq (2005). *On the Finite Element Solution of the Pure Neumann Problem*. SIAM Review 47(1), 50–66. [doi:10.1137/S0036144503426074](https://doi.org/10.1137/S0036144503426074)
3. M. R. Hestenes, E. Stiefel (1952). *Methods of conjugate gradients for solving linear systems*. J. Res. Natl. Bur. Stand. 49(6), 409–436. [doi:10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044)
