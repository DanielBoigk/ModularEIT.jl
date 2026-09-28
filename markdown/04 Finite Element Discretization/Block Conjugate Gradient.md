---
tags: [numerics, linear-algebra, performance]
aliases: [Block CG, BCG, Projected block CG]
---

The **block conjugate gradient** method (O'Leary 1980) solves $A X = B$ for $s$ right-hand sides simultaneously. It minimises the $A$-norm error of every column over the *joint* block Krylov space

$$
\mathcal K_k(A,R_0) = \operatorname{span}\{R_0, AR_0,\dots,A^{k-1}R_0\}\qquad(\text{dimension up to } ks).
$$

In EIT all state (and adjoint) solves of one iteration share the matrix $L_\sigma$, so this is the natural solver (see [[Block Krylov Methods]]).

**Iteration** (Galerkin form, which allows arbitrary bases of the search block):

$$
\begin{aligned}
&P \leftarrow \text{orthonormal basis of } \operatorname{span}(P)\\
&\alpha = (P^\top AP)^{-1}P^\top R,\qquad X \leftarrow X+P\alpha,\qquad R\leftarrow R-AP\alpha,\\
&Z = M^{-1}R,\qquad \beta = -(P^\top AP)^{-1}(AP)^\top Z,\qquad P\leftarrow Z+P\beta .
\end{aligned}
$$

$\beta$ makes the new block $A$-conjugate to the old one. $\alpha$ is the Galerkin solution of the residual equation on the current block.

**Convergence.** The error of each column is bounded like CG with the *reduced* condition number $\kappa_s = \lambda_{\max}/\lambda_s$ instead of $\lambda_{\max}/\lambda_{\min}$: the block effectively deflates the $s-1$ smallest eigenvalues. Block CG therefore needs fewer iterations than single-vector CG, and each iteration does one sparse matrix × *block* product (SpMM) and dense $n\times s\times s$ BLAS-3 operations. These use memory bandwidth and GPU cores far better than $s$ separate sparse matrix–vector products.

**Breakdown and its cure.** If the columns of $P$ become (nearly) linearly dependent, $P^\top AP$ is singular. This happens with dependent right-hand sides, or when some columns converge before others. Robust implementations

- **orthonormalise the search block with rank detection**. SVQB does this: normalise the columns, eigendecompose the $s\times s$ Gram matrix, and drop eigenvalues below a relative threshold. This shrinks the block to its numerical rank, as in Dubrulle's variants;
- **deflate converged columns**: remove their residuals from the next search block.

**Projected version for singular systems.** For the Neumann problem, project every residual and preconditioned residual onto $V^\perp$ and ground at the end, exactly as in [[Projected Conjugate Gradient]]. All $s\times s$ quantities (Gram matrices, the small SPD solves) are tiny and can be handled on the host. All $O(n)$ work is SpMM, GEMM and broadcasting, so the same code runs on CPU and GPU. With a sparse × dense kernel written in KernelAbstractions.jl it is vendor-neutral and runs on NVIDIA, AMD, Intel and Apple GPUs; GPUArrays.jl itself provides no generic sparse × dense product.

**Preconditioning.** A symmetric smoothed-aggregation [[Algebraic Multigrid]] V-cycle with damped Jacobi smoothing, applied to the whole block, makes the iteration count nearly independent of the mesh size. Jacobi smoothing, sparse transfer operators and a dense pseudo-inverse on the coarsest level all run on the GPU. Only the hierarchy setup (aggregation) is done on the CPU.

**CPU or GPU?** A GPU solve has a fixed cost of a few milliseconds (kernel launches and small host round trips each iteration), so it only pays off for enough work. For AMG-preconditioned block CG on an RTX 3080 against an 8-core CPU, the GPU was faster once $n\cdot s\gtrsim 5\cdot10^3$ in single precision or $\gtrsim 2\cdot10^4$ in double precision. At $10^6$ unknowns with 32 right-hand sides it was about 11× (Float64) and 80× (Float32) faster. Consumer GPUs run Float64 at 1/64 of the Float32 rate. Also, tall-skinny Gram products $X^\top Y$ with few columns can hit poorly tuned GEMM kernels, where one matrix–vector product per column was 10–30× faster.

## References

1. D. P. O'Leary (1980). *The block conjugate gradient algorithm and related methods*. Linear Algebra Appl. 29, 293–322. [doi:10.1016/0024-3795(80)90247-5](https://doi.org/10.1016/0024-3795(80)90247-5)
2. A. A. Dubrulle (2001). *Retooling the method of block conjugate gradients*. Electron. Trans. Numer. Anal. 12, 216–233. [etna.ricam.oeaw.ac.at/vol.12.2001/pp216-233.dir/pp216-233.pdf](https://etna.ricam.oeaw.ac.at/vol.12.2001/pp216-233.dir/pp216-233.pdf)
3. A. Stathopoulos, K. Wu (2002). *A Block Orthogonalization Procedure with Constant Synchronization Requirements*. SIAM J. Sci. Comput. 23(6), 2165–2182. [doi:10.1137/S1064827500370883](https://doi.org/10.1137/S1064827500370883)
4. P. Vaněk, J. Mandel, M. Brezina (1996). *Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems*. Computing 56, 179–196. [doi:10.1007/BF02238511](https://doi.org/10.1007/BF02238511)
