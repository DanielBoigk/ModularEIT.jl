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

## Variable conductivity: preconditioning

For $\sigma_{\min}\le\sigma\le\sigma_{\max}$, the energies satisfy $\sigma_{\min}\,u^\top K u\le u^\top L(\sigma)\,u\le\sigma_{\max}\,u^\top K u$. The preconditioned operator $K^\dagger L(\sigma)$ therefore has condition number at most $\sigma_{\max}/\sigma_{\min}$ on the complement of the constants, **independently of the mesh**. The number of [[Projected Conjugate Gradient]] iterations is bounded by $\tfrac12\sqrt{\sigma_{\max}/\sigma_{\min}}\,\ln(2/\varepsilon)$ for a relative tolerance $\varepsilon$ on every mesh, not proportional to $1/h$ as for Jacobi preconditioning. On coarse meshes CG converges earlier, because there are fewer distinct eigenvalues, so the iteration count first grows under refinement and then levels off below the bound. The scale $\bar\sigma$ of the preconditioner $\bar\sigma K$ does not change the iteration count.

Two variants are worth comparing:

1. **Constant coefficient**, $P = \bar\sigma K$. It is robust and depends only on the contrast.
2. **Scaled** (Concus–Golub), $P^{-1} = S^{-1/2}K^\dagger S^{-1/2}$ with the nodal values $S = \operatorname{diag}(\sigma)$ (σ averaged over the cells around each node). Formally, $\nabla\cdot\sigma\nabla u$ becomes $\Delta v-qv$ for $v = \sqrt\sigma\,u$ with $q = \Delta\sqrt\sigma/\sqrt\sigma$. For smooth σ, the condition number then hardly depends on the contrast. At jumps, $q$ behaves like the derivative of a step, its discrete version grows as the mesh is refined, and so does the condition number. The constant variant is the right choice for piecewise constant conductivities.

Nodal conductivities need not be known separately: they can be read off the system matrix as $s_i = L(\sigma)_{ii}/K_{ii}$, a weighted mean of σ around node $i$.

Compared with [[Algebraic Multigrid]], the transform needs no setup that depends on σ. It works unchanged for every σ during a reconstruction and applies to many right-hand sides at once (see [[Block Conjugate Gradient]]).

## Low-rank corrections: contact terms and constrained nodes

Two kinds of modifications of $K$ occur, both supported on few nodes:

- the **contact term** of the complete electrode model, $B = E_B B_r E_B^\top$, on the $r$ boundary nodes under the electrodes;
- **constrained nodes** $D$ of voltage-driven problems, where $u = 0$ is prescribed.

Write $G = (E_B\ \ E_D)$ with the injections $E_B$, $E_D$ of these nodes. The equations $(K+B)\,y = b$ on the free nodes, $y_D = 0$, become, with $\mu = B_r E_B^\top y$ and Lagrange multipliers $\lambda$ for the constraints,

$$
K y+G\nu = b,\qquad G^\top y-\hat C\,\nu = 0,\qquad \nu = \begin{pmatrix}\mu\\ \lambda\end{pmatrix},\quad \hat C = \begin{pmatrix}B_r^{-1} & 0\\ 0 & 0\end{pmatrix}.
$$

$K$ is singular, but the pseudo-inverse is exact on its range. So write $y = K^\dagger(b-G\nu)+c\,\mathbf 1$ and add the solvability condition $\mathbf 1^\top(b-G\nu) = 0$. This gives the bordered **capacitance system**

$$
\begin{pmatrix} G^\top K^\dagger G+\hat C & -G^\top\mathbf 1\\ -\mathbf 1^\top G & 0\end{pmatrix}
\begin{pmatrix}\nu\\ c\end{pmatrix} =
\begin{pmatrix} G^\top K^\dagger b\\ -\mathbf 1^\top b\end{pmatrix}
$$

of size $r+|D|+1$. The matrix $G^\top K^\dagger G$ (entries of the discrete Neumann Green's function between the special nodes) is computed once, from $r+|D|$ transform solves. It depends only on the mesh, the electrodes and the constrained nodes. After a conductivity update only $\hat C$ changes (through $\bar\sigma$ or $S$), so refactoring the small system is cheap. Each application costs two transform solves.

Regularising $K$ instead, e.g. $\tilde K = K+\gamma\,w\,w^\top$ with the trapezoidal weights $w$ (diagonal in the transform basis), and correcting by $-\gamma\,w\,w^\top$ inside the capacitance matrix, is numerically unstable: its diagonal entry $w^\top\tilde K^{-1}w-1/\gamma$ is a difference of two terms of size $1/\gamma$, and $\gamma$ must be of the order of the smallest eigenvalue, $\sim N^{-4}$ in these units.

## Electrode models

- **Continuum, point and gap models**, current-driven: only the right-hand side involves the electrodes, and the operator is exactly $L(\sigma)$, so the transform pseudo-inverse is the preconditioner (see [[Discrete Electrode Models]]).
- **Complete electrode model.** The system

$$
\begin{pmatrix} L(\sigma)+B & A_{uU}\\ A_{uU}^\top & A_{UU}\end{pmatrix}
\begin{pmatrix}u\\ U\end{pmatrix} = \begin{pmatrix}0\\ I\end{pmatrix}
$$

adds the contact term $B = \sum_\ell z_\ell^{-1}M_{\Gamma,e_\ell}$ and $L$ electrode potentials $U$, coupled through constant blocks (see [[Complete Electrode Model]]). The u block $\bar\sigma K+B$ is inverted with the capacitance system above. The electrode potentials are eliminated with the $L\times L$ Schur complement $A_{UU}-A_{uU}^\top(\bar\sigma K+B)^{-1}A_{uU}$, which is recomputed after every conductivity update. It is singular (the constants of $(u,U)$) and is applied as a pseudo-inverse.

## Dirichlet problems

Voltage-driven problems prescribe u on the constrained nodes: all boundary nodes for the continuum model, the electrode nodes for the gap and point models. For the complete electrode model the prescribed electrode voltages only remove $U$, and the u block keeps its contact term. The constrained nodes enter the capacitance system as $E_D$. When the whole boundary is constrained, the interior operator is also diagonalised directly by sines (DST-I). This avoids the $O(N)$ capacitance nodes on very fine grids.

## Implementation notes

- **Grid detection.** The discretization must be a conforming uniform tensor grid of bilinear quadrilaterals, with any spacings $h_x\ne h_y$. The node coordinates give the permutation to lexicographic order. The pixel-aligned meshes of [[Pixel Images and Finite Element Functions]] qualify.
- **Transforms.** DCT-I through the real FFT of the even extension, along both grid directions of an $n_x\times n_y\times s$ block, so the code only needs the generic FFT interface (on GPUs as well).
- **Setup cost.** The capacitance matrix needs one transform solve per contact node and per constrained node. For the complete electrode model that is about the number of boundary nodes under the electrodes, once per forward model.
- **Checks.** For constant σ the preconditioner is the exact (pseudo-)inverse, for all electrode models and for both problem types, so block CG converges in one iteration. For variable σ the preconditioned operator must be symmetric positive definite on the complement of the null space with condition number $\le\sigma_{\max}/\sigma_{\min}$.

**Beyond rectangles.** A disk embedded in a square can be handled with the capacitance matrix method for irregular regions (Buzbee et al.). The correction then acts on all boundary nodes of the disk ($O(\sqrt n)$ of them), and the approximation of the curved boundary by grid lines has to be dealt with separately.

**In ModularEIT.jl:** [`DCTPreconditioner`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.DCTPreconditioner), [`dct_preconditioner`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.dct_preconditioner), [`structured_grid`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.structured_grid).

## References

1. P. Concus, G. H. Golub (1973). *Use of Fast Direct Methods for the Efficient Numerical Solution of Nonseparable Elliptic Equations*. SIAM J. Numer. Anal. 10(6), 1103–1120. [doi:10.1137/0710092](https://doi.org/10.1137/0710092)
2. B. L. Buzbee, F. W. Dorr, J. A. George, G. H. Golub (1971). *The Direct Solution of the Discrete Poisson Equation on Irregular Regions*. SIAM J. Numer. Anal. 8(4), 722–736. [doi:10.1137/0708066](https://doi.org/10.1137/0708066)
3. P. N. Swarztrauber (1977). *The Methods of Cyclic Reduction, Fourier Analysis and the FACR Algorithm for the Discrete Solution of Poisson's Equation on a Rectangle*. SIAM Rev. 19(3), 490–501. [doi:10.1137/1019071](https://doi.org/10.1137/1019071)
4. G. Strang (1999). *The Discrete Cosine Transform*. SIAM Rev. 41(1), 135–147. [doi:10.1137/S0036144598336745](https://doi.org/10.1137/S0036144598336745)
