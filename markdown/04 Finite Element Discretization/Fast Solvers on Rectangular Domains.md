---
tags: [numerics, fem, linear-solvers, preconditioning]
aliases: [DCT preconditioner, Fast Poisson preconditioner, Capacitance matrix method]
---

On a rectangle discretised by a uniform grid (for example the pixel-aligned bilinear meshes of [[Pixel Images and Finite Element Functions]]), the [[Discrete Cosine Transform]] inverts the constant-coefficient Neumann operator in $O(n\log n)$ operations. For the [[Weighted Stiffness Matrix]] $L(\sigma)$ this gives a fast direct solver when σ is constant and a preconditioner with mesh-independent condition number when σ varies. These notes collect the technical points for an implementation.

## Requirements

- Tensor-product grid: $N_x\times N_y$ (or $N_x\times N_y\times N_z$) cells with constant spacings $h_x$, $h_y$ (and $h_z$). The spacings may differ from each other. No hanging nodes.
- Bilinear u space (Q1). The σ space is arbitrary, since only the operator on u is replaced.
- A permutation between the finite element numbering of the u dofs and the lexicographic grid numbering. It is found once from the node coordinates, and the tensor structure is checked at the same time.

## Constant conductivity

For $\sigma\equiv\bar\sigma$, $L = \bar\sigma K_{2D}$, and $K_{2D}^\dagger$ is one forward transform, a diagonal scaling and one backward transform (see [[Discrete Cosine Transform]]). The solution has zero domain integral. The boundary grounding of the library (see [[Grounding of the Potential]]) is restored by subtracting a constant.

The zero eigenvalue can also be replaced instead of dropped. With the trapezoidal weights $w$,

$$
\tilde K = K_{2D}+\gamma\,w\,w^\top
$$

is diagonal in the same basis: only the eigenvalue of the constant mode changes, from $0$ to $\gamma\,d_0$. $\tilde K$ is nonsingular, which the low-rank corrections below need.

## Variable conductivity: preconditioning

For $\sigma_{\min}\le\sigma\le\sigma_{\max}$, the energies satisfy $\sigma_{\min}\,u^\top K u\le u^\top L(\sigma)\,u\le\sigma_{\max}\,u^\top K u$. The preconditioned operator $K^\dagger L(\sigma)$ therefore has condition number at most $\sigma_{\max}/\sigma_{\min}$ on the complement of the constants, **independently of the mesh**. The number of [[Projected Conjugate Gradient]] iterations grows like $\sqrt{\sigma_{\max}/\sigma_{\min}}$, not like $1/h$ as for Jacobi preconditioning. The scale $\bar\sigma$ of the preconditioner $\bar\sigma K$ does not change the iteration count.

Two variants are worth comparing:

1. **Constant coefficient**, $P = \bar\sigma K$. It is robust and depends only on the contrast.
2. **Scaled** (Concus–Golub), $P^{-1} = S^{-1/2}K^\dagger S^{-1/2}$ with the nodal values $S = \operatorname{diag}(\sigma)$ (piecewise constant σ averaged over the cells around each node). Formally, $\nabla\cdot\sigma\nabla u$ becomes $\Delta v-qv$ for $v = \sqrt\sigma\,u$ with $q = \Delta\sqrt\sigma/\sqrt\sigma$. For smooth σ, the condition number then no longer depends on the contrast. At jumps $q$ is large and the scaled variant loses this advantage.

Compared with [[Algebraic Multigrid]], the transform needs no setup that depends on σ. It works unchanged for every σ during a reconstruction and applies to many right-hand sides at once (see [[Block Conjugate Gradient]]).

## Electrode models

- **Continuum, point and gap models** are Neumann problems. Only the right-hand side involves the electrodes, and the operator is exactly $L(\sigma)$ (see [[Discrete Electrode Models]]).
- **Complete electrode model.** The system

$$
\begin{pmatrix} L(\sigma)+B & -C\\ -C^\top & G\end{pmatrix}
\begin{pmatrix}u\\ U\end{pmatrix} = \begin{pmatrix}0\\ I\end{pmatrix}
$$

adds the contact term $B = \sum_\ell z_\ell^{-1}M_{\Gamma,e_\ell}$, which is supported on the $r$ boundary nodes under the electrodes, and $L$ electrode potentials $U$ (see [[Complete Electrode Model]]). Write $B = E\,B_r E^\top$, with $E$ the injection of the electrode nodes. The preconditioner for the u block is $\bar\sigma\tilde K+B$, applied with the Woodbury identity (the capacitance matrix method):

$$
(\bar\sigma\tilde K+E B_r E^\top)^{-1} = \tilde K_\sigma^{-1}-\tilde K_\sigma^{-1}E\,\big(B_r^{-1}+E^\top\tilde K_\sigma^{-1}E\big)^{-1}E^\top\tilde K_\sigma^{-1},\qquad \tilde K_\sigma = \bar\sigma\tilde K .
$$

The $r\times r$ capacitance matrix $E^\top\tilde K^{-1}E$ is computed once, from $r$ transform solves. It depends only on the mesh and the electrodes, not on σ, and $\bar\sigma$ enters as a scalar factor. The $L\times L$ Schur complement for $U$ is then small and dense. If $z_\ell$ is large compared with $\bar\sigma h$, $B$ can simply be left out of the preconditioner.

## Dirichlet problems

Voltage-driven problems with Dirichlet data on the whole boundary act on the interior nodes. There the constant-coefficient operator is diagonalised by sines (DST-I) and has no null space. Dirichlet data on parts of the boundary are handled like the contact term: take the Neumann transform and apply a Woodbury correction on the constrained nodes.

## Implementation plan

1. **Grid detection.** A helper checks that a discretization is a uniform tensor grid and returns $N_x$, $N_y$, $h_x$, $h_y$ and the permutation to lexicographic order. [[Pixel Images and Finite Element Functions|Image maps]] on pixel-aligned meshes already rely on the same structure.
2. **Transforms.** Implement DCT-I, DCT-II and DST-I with even or odd extensions and a complex FFT through the AbstractFFTs interface. FFTW is the CPU back end. On GPUs the FFT library plans on device arrays, so the same code runs there. Transforms act along the first dimensions of an $n\times s$ block.
3. **Preconditioner.** A `DCTPreconditioner` implements `apply_preconditioner!` for [[Projected Conjugate Gradient|projected block CG]], with the `:constant` and `:scaled` variants and an optional capacitance correction for the complete electrode model. It becomes a solver choice (`BlockCGSolver(preconditioner = :dct)`).
4. **Tests.** Exact solve (one iteration) for constant σ. Iteration counts constant under uniform refinement. The contrast bound. Equality of the Woodbury and assembled complete electrode model preconditioners on small meshes. JLArrays with scalar indexing disabled.
5. **Benchmarks** (on request): against AMG and the sparse Cholesky factorisation, over mesh size, contrast and the number of right-hand sides, on CPU and GPU.

**Beyond rectangles.** A disk embedded in a square can be handled with the capacitance matrix method for irregular regions (Buzbee et al.). The correction then acts on all boundary nodes of the disk ($O(\sqrt n)$ of them), and the approximation of the curved boundary by grid lines has to be dealt with separately.

## References

1. P. Concus, G. H. Golub (1973). *Use of Fast Direct Methods for the Efficient Numerical Solution of Nonseparable Elliptic Equations*. SIAM J. Numer. Anal. 10(6), 1103–1120. [doi:10.1137/0710092](https://doi.org/10.1137/0710092)
2. B. L. Buzbee, F. W. Dorr, J. A. George, G. H. Golub (1971). *The Direct Solution of the Discrete Poisson Equation on Irregular Regions*. SIAM J. Numer. Anal. 8(4), 722–736. [doi:10.1137/0708066](https://doi.org/10.1137/0708066)
3. P. N. Swarztrauber (1977). *The Methods of Cyclic Reduction, Fourier Analysis and the FACR Algorithm for the Discrete Solution of Poisson's Equation on a Rectangle*. SIAM Rev. 19(3), 490–501. [doi:10.1137/1019071](https://doi.org/10.1137/1019071)
4. G. Strang (1999). *The Discrete Cosine Transform*. SIAM Rev. 41(1), 135–147. [doi:10.1137/S0036144598336745](https://doi.org/10.1137/S0036144598336745)
