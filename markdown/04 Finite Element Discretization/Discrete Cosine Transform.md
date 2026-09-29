---
tags: [numerics, fem, fft]
aliases: [DCT, DCT-I, DCT-II, Fast cosine transform]
---

On a uniform grid, the finite element matrices of Neumann problems are diagonalised by **discrete cosine transforms (DCT)**. This is the basis of fast solvers and spectral norms on rectangles (see [[Fast Solvers on Rectangular Domains]] and [[Spectral Sobolev Norms on Rectangles]]).

## Two variants

Different grid positions give different cosine bases.

- **DCT-I (nodal).** For the $N+1$ nodes $x_j = jh$, $j = 0,\dots,N$, of a uniform 1D mesh, the modes are
$$
v_k(j) = \cos\frac{\pi k j}{N},\qquad k = 0,\dots,N .
$$
- **DCT-II (cell-centred).** For the $n$ cell centres $x_j = (j+\tfrac12)h$, $j = 0,\dots,n-1$, the modes are
$$
c_k(j) = \cos\frac{\pi k\,(j+\frac12)}{n},\qquad k = 0,\dots,n-1 .
$$

Both are discrete versions of the Neumann eigenfunctions $\cos(\pi k x/\ell)$ of $-\mathrm d^2/\mathrm dx^2$ on an interval of length $\ell$.

## Linear elements: nodal matrices

For linear elements with spacing $h$, the 1D Neumann [[Stiffness Matrix]] $K$ and [[Mass Matrix]] $M$ are tridiagonal, with halved diagonal entries in the first and last rows. Let $V = (v_k(j))_{j,k}$ and $W = \operatorname{diag}(\tfrac12,1,\dots,1,\tfrac12)$ (trapezoidal weights). With $\theta_k = \pi k/N$,

$$
K\,V = W\,V\,\Lambda_K,\qquad M\,V = W\,V\,\Lambda_M,
$$
$$
\Lambda_K = \operatorname{diag}\Big(\frac2h\,(1-\cos\theta_k)\Big),\qquad
\Lambda_M = \operatorname{diag}\Big(\frac h6\,(4+2\cos\theta_k)\Big).
$$

The interior rows are the standard identity $2\cos\theta j-\cos\theta(j-1)-\cos\theta(j+1) = 2(1-\cos\theta)\cos\theta j$. In the boundary rows both sides carry the factor $\tfrac12$ of $W$. The modes are orthogonal in the trapezoidal inner product,

$$
V^\top W\,V = D = \operatorname{diag}(d_k),\qquad d_0 = d_N = N,\quad d_k = \tfrac N2\ \text{otherwise},
$$

so $V^{-1} = D^{-1}V^\top W$. Consequently $K = W\,V\,\Lambda_K D^{-1}V^\top W$ and $K\,V = M\,V\,\Lambda_M^{-1}\Lambda_K$: the columns of $V$ solve the generalised eigenproblem of the pair $(K, M)$.

**Tensor products.** On a uniform $N_x\times N_y$ rectangle grid with spacings $h_x, h_y$, the bilinear (Q1) matrices are Kronecker products of 1D matrices,

$$
K_{2D} = K_x\otimes M_y+M_x\otimes K_y,\qquad M_{2D} = M_x\otimes M_y ,
$$

so $V_x\otimes V_y$ diagonalises them, with eigenvalues $\lambda^K_k\lambda^M_l+\lambda^M_k\lambda^K_l$ (spacing $h_x$ in the first, $h_y$ in the second factor of each product). The same holds for three factors in 3D.

**Solving.** The only zero eigenvalue belongs to the constant mode $k = l = 0$ (see [[Null Space of the Neumann Problem]]). Dropping it gives

$$
K_{2D}^{\dagger} = (V_x\otimes V_y)\,\Lambda^{\dagger}(D_x\otimes D_y)^{-1}(V_x\otimes V_y)^\top ,
$$

whose solutions satisfy $\sum_j w_j u_j = 0$ with the tensor trapezoidal weights $w = W_x\otimes W_y$. For bilinear functions this is exactly $\int_\Omega u\,\mathrm dx = 0$, so the transform solver comes with its own grounding; other groundings follow by subtracting a constant (see [[Grounding of the Potential]]).

## Piecewise constants: cell-centred matrices

For piecewise constants on the cells of the same grid, the two-point-flux jump matrix $\sum_F \frac{\vert F\vert}{d_F}(\sigma_K-\sigma_{K'})^2$ is, in 1D, $L = \operatorname{tridiag}(-1,2,-1)$ with ones in the corners. It satisfies

$$
L\,C = C\operatorname{diag}\big(2(1-\cos(\pi k/n))\big),\qquad C = (c_k(j))_{j,k},
$$

and $C^\top C$ is diagonal. In 2D the jump matrix is $\tfrac{h_y}{h_x}L_x\otimes I+\tfrac{h_x}{h_y}I\otimes L_y$, diagonalised by $C_x\otimes C_y$. The mass matrix is $h_x h_y I$.

## Dirichlet conditions

On the interior nodes (Dirichlet data eliminated, see [[Enforcing Dirichlet Conditions]]), the modes are sines $\sin(\pi k j/N)$, $k = 1,\dots,N-1$ (DST-I). The eigenvalues are the same expressions in $\theta_k$, and $W = I$.

## Fast evaluation

Products with $V$ and $V^\top W$ cost $O(N\log N)$ through the FFT. The even extension $y = (x_0,\dots,x_N,x_{N-1},\dots,x_1)$ of length $2N$ gives

$$
\operatorname{Re}\,\operatorname{FFT}(y)_k = 2\,\big(V^\top W\,x\big)_k ,
$$

and $V$ is symmetric, so $V c = (V^\top W)(W^{-1}c)$ uses the same routine. DCT-II and DST-I follow from analogous extensions, or from one FFT of length $n$ after reordering (Makhoul). A complex FFT is therefore sufficient, which matters on GPUs, where FFT libraries do not provide cosine transforms. Several right-hand sides are transformed as one batch.

**In ModularEIT.jl:** [`dct_neumann_solve`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.dct_neumann_solve), [`StructuredGrid`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.StructuredGrid).

## References

1. G. Strang (1999). *The Discrete Cosine Transform*. SIAM Rev. 41(1), 135–147. [doi:10.1137/S0036144598336745](https://doi.org/10.1137/S0036144598336745)
2. J. Makhoul (1980). *A fast cosine transform in one and two dimensions*. IEEE Trans. Acoust. Speech Signal Process. 28(1), 27–34. [doi:10.1109/TASSP.1980.1163351](https://doi.org/10.1109/TASSP.1980.1163351)
3. P. N. Swarztrauber (1977). *The Methods of Cyclic Reduction, Fourier Analysis and the FACR Algorithm for the Discrete Solution of Poisson's Equation on a Rectangle*. SIAM Rev. 19(3), 490–501. [doi:10.1137/1019071](https://doi.org/10.1137/1019071)
