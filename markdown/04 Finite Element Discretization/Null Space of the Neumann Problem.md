---
tags: [numerics, fem]
aliases: [Grounding, Pinning]
---

The discrete [[Neumann Problem]] $L_\gamma\mathbf u = \mathbf g$ is singular: $L_\gamma\mathbf 1 = 0$. It is solvable iff $\mathbf g\perp\mathbf 1$, that is, $\sum_i g_i = \int_{\partial\Omega} g = 0$. Then the solution is unique up to adding a constant. Ways to handle this:

1. **Project the data.** Remove the mean of the current pattern, $\mathbf g\leftarrow\mathbf g - \frac{\mathbf 1^\top\mathbf g}{\mathbf 1^\top\mathbf 1}\mathbf 1$ (or the $M_\Gamma$-weighted version). Without this, the system is inconsistent, and a regularised solver silently returns the solution to a different problem.
2. **Krylov on the singular system.** CG and MINRES converge on consistent singular symmetric systems if started in $\operatorname{range}(L)$. Projecting the iterates and the preconditioner output onto $\mathbf 1^\perp$ keeps round-off from drifting into the kernel. MINRES handles semidefinite and nearly singular cases more robustly.
3. **Pinning.** Fix one DOF, $u_k = 0$, by replacing its row and column as for [[Enforcing Dirichlet Conditions|a Dirichlet condition]]. This gives an SPD matrix and is simple, but the grounding is attached to an arbitrary node. The solution is re-grounded afterwards.
4. **Mean-value constraint.** Add a Lagrange multiplier for $\int_{\partial\Omega}u = 0$ (or $\int_\Omega u = 0$), which gives a saddle-point system of size $n+1$.
5. **Shift $L+\varepsilon I$.** Makes the matrix SPD, but it solves a slightly different problem. The relative perturbation of the solution is of order $\varepsilon/\lambda_2$, where $\lambda_2$ is the smallest nonzero eigenvalue of $L$. It is therefore used with very small $\varepsilon$, together with the projection in 1.

After solving, **ground** the solution by subtracting its boundary mean, so voltages are comparable to measurements that are themselves mean-free.

## References

1. P. Bochev, R. B. Lehoucq (2005). *On the Finite Element Solution of the Pure Neumann Problem*. SIAM Review 47(1), 50–66. [doi:10.1137/S0036144503426074](https://doi.org/10.1137/S0036144503426074)
2. C. C. Paige, M. A. Saunders (1975). *Solution of Sparse Indefinite Systems of Linear Equations*. SIAM J. Numer. Anal. 12(4), 617–629. [doi:10.1137/0712047](https://doi.org/10.1137/0712047)
