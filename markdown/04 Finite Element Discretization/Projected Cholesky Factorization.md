---
tags: [numerics, linear-algebra]
aliases: [Projected Cholesky, Cholesky for singular systems, Pinned Cholesky]
---

Direct solver for $A\mathbf x = \mathbf b$ when $A$ is symmetric positive **semi**definite with a known null space $V = \ker A$ of dimension $k$ and positive definite on $V^\perp$. Examples are the pure Neumann [[Weighted Stiffness Matrix]], with $V = \operatorname{span}\{\mathbf 1\}$ and $k = 1$, or elasticity with rigid-body modes. A plain Cholesky factorisation breaks down on the zero pivot. Adding a dense correction such as $A + c\,\mathbf 1\mathbf 1^\top$ would make it nonsingular, but would destroy the sparsity.

**Pinning.** Choose $k$ indices $I$ such that the $k\times k$ block $V_I$ (rows $I$ of a basis matrix) is invertible, and let $J$ be the complement. For constants any single index works. For a general $V$, a column-pivoted QR factorisation of $V^\top$ picks a well-conditioned choice.

> **Proposition.** $A_{JJ}$ is symmetric positive definite.
>
> *Proof.* Let $\mathbf y\in\mathbb R^{|J|}$ with $\mathbf y^\top A_{JJ}\mathbf y = 0$, and extend it by zeros on $I$ to $\mathbf z$. Then $\mathbf z^\top A\mathbf z = 0$, so $\mathbf z\in\ker A$, i.e. $\mathbf z = V\mathbf c$. On $I$ this gives $V_I\mathbf c = \mathbf z_I = 0$, hence $\mathbf c = 0$ and $\mathbf y = 0$. $\square$

$A_{JJ}$ is a principal submatrix of $A$, so it is exactly as sparse and has an ordinary sparse Cholesky factorisation $A_{JJ} = LL^\top$ with the usual fill-reducing orderings (AMD, nested dissection).

**Solving.** For a right-hand side $\mathbf b$:

1. make it consistent: $\hat{\mathbf b} = \Pi\mathbf b$ with $\Pi = I - VV^\top$ ($V$ orthonormal). For $V = \operatorname{span}\{\mathbf 1\}$ this subtracts the mean;
2. solve $A_{JJ}\mathbf x_J = \hat{\mathbf b}_J$ and set $\mathbf x_I = 0$;
3. ground the solution: $\mathbf x \leftarrow \mathbf x - V(W^\top V)^{-1}W^\top\mathbf x$ (see [[Grounding of the Potential]]).

The vector from step 2 solves $A\mathbf x = \hat{\mathbf b}$. Take any solution $\mathbf x^*$ and shift it along $V$ so that it vanishes on $I$: $\tilde{\mathbf x} = \mathbf x^* - VV_I^{-1}\mathbf x^*_I$. This is still a solution, and its $J$-rows satisfy the same nonsingular system $A_{JJ}\tilde{\mathbf x}_J = \hat{\mathbf b}_J$, so $\tilde{\mathbf x}_J = \mathbf x_J$. Step 3 then moves to the representative required by the grounding condition, for example zero sum over the boundary nodes. The equations of the pinned rows never have to be solved: they hold automatically because $\hat{\mathbf b}\perp V$.

**Block right-hand sides.** Triangular solves with $L$ and $L^\top$ handle $s$ right-hand sides at once (BLAS-3 in supernodal codes). In EIT, all state *and* adjoint solves of one reconstruction iteration share one factorisation: $2N$ solves per factorisation.

**Changing conductivity.** A new $\sigma$ changes the values of $L_\sigma$ but not its sparsity pattern. The symbolic analysis (ordering, elimination tree, supernode structure) is reused and only the numeric factorisation is repeated. Keeping a map from the entries of $A$ to those of $A_{JJ}$ makes this update a simple gather.

**Cost.** For 2D finite element meshes with $n$ unknowns, nested dissection gives $\mathcal O(n\log n)$ fill and $\mathcal O(n^{3/2})$ factorisation work. Each solve costs $\mathcal O(n\log n)$ per right-hand side. Compared with [[Block Conjugate Gradient|(block) CG]] with [[Algebraic Multigrid|AMG]], this pays off when many right-hand sides share one matrix, when high accuracy is needed, or for moderate $n$. For very large 3D problems the fill-in makes iterative solvers preferable.

*Measured (ModularEIT.jl, 2D Q1 mesh, 32 right-hand sides, RTX 3080 / Ryzen 7 7800X3D):* at $10^6$ unknowns, re-factorisation plus solve takes 2.2 s with CHOLMOD on the CPU and 0.19 s with cuDSS on the GPU (Float64). AMG-preconditioned block CG needs 20.5 s and 1.8 s respectively. **Single precision is not safe for the direct solver:** the condition number of the stiffness matrix grows like $h^{-2}$, and Float32 factorisations reach only about $10^{-2}$ relative residual at $10^6$ unknowns, while Float32 CG stays near $10^{-4}$.

**Alternatives.** The bordered (Lagrange multiplier) system
$\begin{pmatrix}A & V\\ V^\top & 0\end{pmatrix}$
is nonsingular but indefinite, so it needs an $LDL^\top$ factorisation. The shift $A+\varepsilon I$ changes the solution (see [[Null Space of the Neumann Problem]]). Pinning keeps the SPD structure and gives the exact solution.

## References

1. P. Bochev, R. B. Lehoucq (2005). *On the Finite Element Solution of the Pure Neumann Problem*. SIAM Review 47(1), 50–66. [doi:10.1137/S0036144503426074](https://doi.org/10.1137/S0036144503426074)
2. Y. Chen, T. A. Davis, W. W. Hager, S. Rajamanickam (2008). *Algorithm 887: CHOLMOD, Supernodal Sparse Cholesky Factorization and Update/Downdate*. ACM Trans. Math. Softw. 35(3), 22. [doi:10.1145/1391989.1391995](https://doi.org/10.1145/1391989.1391995)
3. A. George (1973). *Nested Dissection of a Regular Finite Element Mesh*. SIAM J. Numer. Anal. 10(2), 345–363. [doi:10.1137/0710032](https://doi.org/10.1137/0710032)
4. P. R. Amestoy, T. A. Davis, I. S. Duff (1996). *An Approximate Minimum Degree Ordering Algorithm*. SIAM J. Matrix Anal. Appl. 17(4), 886–905. [doi:10.1137/S0895479894278952](https://doi.org/10.1137/S0895479894278952)
5. NVIDIA. *cuDSS: CUDA Direct Sparse Solver* (documentation). [docs.nvidia.com/cuda/cudss](https://docs.nvidia.com/cuda/cudss/)
