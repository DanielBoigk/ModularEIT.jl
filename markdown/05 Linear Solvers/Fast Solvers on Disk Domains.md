---
tags: [numerics, fem, linear-solvers, preconditioning, fft]
aliases: [Polar fast solver, FFT preconditioner on the disk, Block-circulant stiffness matrix]
---

On a disk, the constant-coefficient Neumann problem separates in polar coordinates: Fourier modes in the angle, and one ordinary differential equation along the radius per mode. The finite element version of this needs no special elements. It only needs a mesh that is invariant under a rotation, and it gives a fast solver and a preconditioner for the [[Weighted Stiffness Matrix]] $L(\sigma)$, like the transform methods of [[Fast Solvers on Rectangular Domains]].

## Rotationally symmetric meshes

Take a centre node and $n_r$ rings of $n_\theta$ nodes at radii $0<r_1<\dots<r_{n_r} = R$ and angles $\theta_j = 2\pi j/n_\theta$. Connect the centre to the first ring by a fan of triangles and split every quadrilateral between two rings along the same diagonal. The mesh is then invariant under the rotation $\mathcal R$ by $2\pi/n_\theta$, which maps node $(k,j)$ to $(k,j+1)$. The radii are arbitrary: they can be graded towards the boundary, where EIT sensitivity and the singularities at electrode edges are concentrated (see [[Decay of Boundary Measurements]], [[Complete Electrode Model]]). For $L$ electrodes covering the fraction $c$ of the boundary, $n_\theta$ a multiple of $2L/c$ puts the electrode edges on nodes.

With the same number of nodes on every ring, elements become long and thin near the centre and, with boundary grading, near the boundary. For linear elements this is harmless for the approximation, as long as no angle approaches $\pi$. It does make the matrix strongly anisotropic, and that degrades [[Algebraic Multigrid]]. The transform solver below is exact for any radii and is not affected.

## Block-circulant structure

Order the unknowns ring by ring, $U = (U_{k,j})$. Invariance under $\mathcal R$ means that the stiffness matrix $K$ (constant conductivity) couples $(k,j)$ and $(k',j')$ only through $k$, $k'$ and the offset $\delta = j'-j \bmod n_\theta$:

$$
(K U)_{\cdot,j} = \sum_\delta K_\delta\,U_{\cdot,j+\delta}\qquad(+\ \text{centre coupling}),
$$

with $n_r\times n_r$ blocks $K_\delta$, $K_{-\delta} = K_\delta^\top$. Only neighbouring angles couple, so $\delta\in\{-1,0,1\}$, and only neighbouring rings, so every $K_\delta$ is tridiagonal. The discrete Fourier transform in $j$,

$$
\hat U_m = \sum_{j=0}^{n_\theta-1} U_{\cdot,j}\,e^{-2\pi i jm/n_\theta},
$$

block-diagonalises $K$, as for every circulant matrix:

$$
\hat K_m\,\hat U_m = \hat b_m,\qquad \hat K_m = \sum_\delta K_\delta\,e^{2\pi i\delta m/n_\theta},
$$

a Hermitian tridiagonal system for each angular mode $m$. For real data the modes $m$ and $n_\theta-m$ are complex conjugate, so $m = 0,\dots,\lfloor n_\theta/2\rfloor$ suffice (real FFT). A solve costs one FFT per ring, $O(n_r)$ per mode and one inverse FFT: $O(n\log n_\theta)$ in total.

## The centre node and the constants

The centre node couples with the same weight $\kappa_k$ to all nodes of a ring, so it enters only mode $0$. Mode $0$ also contains the null space of the Neumann problem, the constants (see [[Null Space of the Neumann Problem]]). It is solved together with the centre value and the grounding $\sum_i u_i = 0$ as a bordered system:

$$
\begin{pmatrix} K_{cc} & \kappa^\top & 1\\ n_\theta\,\kappa & \hat K_0 & n_\theta\mathbf 1\\ 1 & \mathbf 1^\top & 0\end{pmatrix}
\begin{pmatrix} u_c\\ \hat U_0\\ c\end{pmatrix} =
\begin{pmatrix} b_c\\ \hat b_0\\ 0\end{pmatrix}.
$$

The result is the Moore–Penrose pseudo-inverse $K^\dagger$: the mean-zero solution for data orthogonal to the constants. Other groundings follow by subtracting a constant (see [[Grounding of the Potential]]).

## Preconditioning, electrodes and Dirichlet problems

Everything else carries over from [[Fast Solvers on Rectangular Domains]], which needs only a fast $K^\dagger$ that is exact on the range of $K$:

- **Variable conductivity.** $\bar\sigma K$ is a preconditioner with condition number at most $\sigma_{\max}/\sigma_{\min}$ on every mesh.
- **Contact terms and constrained nodes.** The contact terms of the complete electrode model and the constrained nodes of voltage-driven problems enter through the bordered capacitance system.
- **Electrode voltages.** These are eliminated with a small Schur complement.

The electrodes need not respect the rotational symmetry: only $K$ is required to be circulant.

The blocks $K_\delta$, $K_{cc}$ and $\kappa$ can be read off the assembled stiffness matrix. Every entry must then agree with its rotated copies, which verifies the symmetry of a given mesh at the same time.

## General domains

The disk is the model domain for every simply connected planar domain. A conformal map $\Phi$ from the disk onto the domain keeps the conductivity equation isotropic in two dimensions (see [[Conformal Invariance of the Conductivity Equation]]). Mapping the nodes of a polar mesh by $\Phi$ gives a mesh of the domain with the same connectivity (see [[Numerical Conformal Mapping]]). Its stiffness matrix is spectrally equivalent to the one of the disk mesh, with constants that tend to $1$ under refinement. The disk solver is therefore an equally good preconditioner there, while the problem itself (electrodes, conductivity, data) is posed on the physical domain.

**In ModularEIT.jl:** [`PolarPreconditioner`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.PolarPreconditioner), [`polar_preconditioner`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.polar_preconditioner), [`polar_grid`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.polar_grid), [`fast_neumann_solve`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.fast_neumann_solve).

## References

1. M.-C. Lai, W.-C. Wang (2001). *Fast direct solvers for Poisson equation on 2D polar and spherical geometries*. Numer. Methods Partial Differ. Equ. 18(1), 56–68. [doi:10.1002/num.1038](https://doi.org/10.1002/num.1038)
2. P. N. Swarztrauber (1974). *A Direct Method for the Discrete Solution of Separable Elliptic Equations*. SIAM J. Numer. Anal. 11(6), 1136–1150. [doi:10.1137/0711086](https://doi.org/10.1137/0711086)
3. T. F. Chan (1988). *An Optimal Circulant Preconditioner for Toeplitz Systems*. SIAM J. Sci. Stat. Comput. 9(4), 766–771. [doi:10.1137/0909051](https://doi.org/10.1137/0909051)
