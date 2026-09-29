---
tags: [numerics, geometry, fft]
aliases: [Theodorsen's method, Wegmann's method, Symm's integral equation, Riemann map, Crowding, Conformally mapped meshes]
---

By the Riemann mapping theorem, every simply connected domain $\Omega\subsetneq\mathbb C$ is the image of the unit disk $D$ under a conformal map $\Phi$. It is unique once $\Phi(0) = z_0$ and $\Phi'(0)>0$ are fixed. Such maps turn EIT on $\Omega$ into EIT on the disk (see [[Conformal Invariance of the Conductivity Equation]]). In practice they have to be computed numerically.

## Theodorsen's method

Suppose the boundary is star-shaped with respect to $z_0$ and given in polar form, $z = z_0+\rho(\vartheta)\,e^{i\vartheta}$. Write

$$
\Phi(w) = z_0+w\,e^{F(w)}
$$

with $F$ holomorphic in the disk. On $|w| = 1$, $w = e^{i\theta}$, the boundary condition says $\Phi(e^{i\theta}) = z_0+\rho(\vartheta)e^{i\vartheta}$ for the unknown **boundary correspondence** $\vartheta(\theta)$:

$$
\operatorname{Re}F(e^{i\theta}) = \log\rho(\vartheta(\theta)),\qquad \operatorname{Im}F(e^{i\theta}) = \vartheta(\theta)-\theta .
$$

The imaginary part of a holomorphic function on the circle is the conjugate function $\mathcal K$ of its real part (with zero mean for $\Phi'(0)>0$). This gives **Theodorsen's integral equation**

$$
\vartheta = \theta+\mathcal K\big[\log\rho\circ\vartheta\big],\qquad \mathcal K:\ e^{ik\theta}\mapsto -i\operatorname{sign}(k)\,e^{ik\theta}.
$$

At $N$ equispaced angles, $\mathcal K$ is one FFT, a multiplication and an inverse FFT, so the fixed-point iteration costs $O(N\log N)$ per step. The Fourier coefficients of the converged $\log\rho\circ\vartheta$ are the Taylor coefficients of $F$ ($a_0 = \hat g_0$, $a_k = 2\hat g_k$). So $\Phi$ is holomorphic in the disk by construction, and its derivative $\Phi'(w) = e^{F}(1+wF')$ is available in closed form.

**Convergence.** The iteration contracts if the boundary is *$\varepsilon$-nearly circular*, $|\mathrm d\log\rho/\mathrm d\vartheta|\le\varepsilon<1$; the error decreases like $\varepsilon^k$. For an ellipse with semi-axes $a>b$, $\varepsilon = (a^2-b^2)/(2ab)$, so aspect ratios up to about $2.4$ are admissible. Thorax- and head-shaped cross-sections are well within the range. Under-relaxation widens it somewhat. Domains that are far from circular, or not star-shaped, need the methods below.

**Accuracy.** The discrete map matches the boundary exactly at the $N$ nodes. Between the nodes the error is the trigonometric interpolation error of $\log\rho\circ\vartheta$. It is spectrally small for smooth boundaries. At corners the boundary correspondence has singular derivatives, and convergence in $N$ is algebraic.

## Symm's integral equation

For any smooth Jordan curve $\eta(s)$, $s\in[0,2\pi)$, the boundary correspondence can be obtained from a **linear** problem. Let $w(s)$ be the density, with respect to the parameter, of the harmonic measure of $\Omega$ at $z_0$. For $z$ outside $\Omega$ the function $\zeta\mapsto\log|z-\zeta|$ is harmonic in $\Omega$, so by the mean value property for harmonic measure

$$
\int_0^{2\pi}\log|\eta(s)-\eta(\tau)|\,w(\tau)\,\mathrm d\tau = \log|\eta(s)-z_0|,\qquad \int_0^{2\pi}w\,\mathrm d\tau = 1,
$$

which holds on the boundary by continuity. Because the Riemann map carries harmonic measure to the uniform measure on the circle, the correspondence is

$$
t(s) = t_0+2\pi\int_0^s w(\tau)\,\mathrm d\tau ,
$$

monotone by construction; $t_0$ is fixed by $\Phi'(0)>0$. The first-kind equation becomes singular when the logarithmic capacity of the curve is $1$. Adding an unknown constant to the left-hand side, which vanishes for the exact solution, together with the side condition gives a nonsingular bordered system. With Kress's quadrature for the logarithmic singularity, $\int\log|2\sin\frac{s-\tau}2|\,\varphi(\tau)\,\mathrm d\tau$ integrated exactly for trigonometric interpolants, the Nyström discretisation is spectrally accurate for smooth curves. It costs a dense $N\times N$ solve.

## Wegmann's method

Wegmann's method solves the nonlinear problem directly by Newton's method. Write $y = \eta(S)-z_0$ and $d = \eta'(S)$ for a current correspondence $S(t)$. The corrected boundary values $y+d\,U$, $U$ real, must equal $e^{it}G$ with $G$ holomorphic and $G(0)>0$. Equivalently,

$$
\operatorname{Im}(a\,G) = b,\qquad a = \frac{e^{it}}{\eta'(S)},\quad b = \operatorname{Im}\frac{y}{d},
$$

a linear **Riemann–Hilbert problem** of index $0$ ($e^{it}$ and $\eta'(S(t))$ both wind once). With $\theta = \arg a$, the function $Q = \exp(-\mathcal K\theta+i\theta)$ is holomorphic with $\arg Q = \theta$, so $a = |a|\,e^{\mathcal K\theta}Q$. The problem reduces to $\operatorname{Im}(QG) = c := b/(|a|e^{\mathcal K\theta})$, with solution

$$
QG = -\mathcal Kc+i\,c+\lambda,\qquad \lambda\in\mathbb R \text{ from } \operatorname{Im}G(0) = 0 .
$$

Then $U = \operatorname{Re}\big((e^{it}G-y)/d\big)$ and $S\leftarrow S+U$. A step costs a few FFTs. The iteration converges quadratically, but only locally. Symm's solution is an excellent starting point; an arc-length correspondence is generally not.

The discrete Newton map needs two safeguards:

- **Oversampling.** The coefficient $\arg a$ varies much faster than the map itself where the correspondence is compressed. The Riemann–Hilbert step is therefore computed on an oversampled grid, for example $4N$ points.
- **Filtering.** The discrete iteration amplifies high-frequency round-off and aliasing, so that without a filter errors grow from step to step even at the exact solution. A low-pass filter on the correction $U$ removes this instability.

## Crowding

For elongated domains $|\Phi'|$ becomes exponentially small on parts of the boundary. For a rectangle of aspect ratio $L$, the harmonic measure of an end, seen from the centre, decays like $e^{-\pi L/2}$. For an ellipse with aspect ratio $3$, $|\Phi'|$ already varies by a factor of several hundred along the boundary. The correspondence then needs thousands of Fourier modes, and mapped grids concentrate their nodes where the domain is wide. The disk is then a poor model domain; a rectangle or strip, with Schwarz–Christoffel-type maps, is the natural choice. Too few modes do not make the iteration fail: it converges to a band-limited map that misses the boundary. The boundary error between the nodes must therefore be monitored, and the resolution increased until it is small.

## Conformally mapped meshes

Mapping the nodes of a disk mesh, such as a graded polar mesh (see [[Fast Solvers on Disk Domains]]), by $\Phi$ gives a mesh of $\Omega$ with the same connectivity:

- **Shape.** Each triangle is distorted only by the variation of $\Phi'$ over the element, which is $O(h)$. Shapes, and a grading towards the boundary, carry over, with the local size scaled by $|\Phi'|$.
- **Conditioning.** The stiffness matrix of the mapped mesh is spectrally equivalent to the one of the disk mesh, with constants $1+O(h\,|\Phi''|/|\Phi'|)$. The fast disk solver is therefore a preconditioner for the mapped problem, and its quality does not deteriorate under refinement.
- **Everything else is standard.** Electrodes, conductivities, data and reconstructions live on the mapped mesh. No boundary weights $|\Phi'|$ are needed, because the finite element method is applied on $\Omega$ itself.

**In ModularEIT.jl:** [`ConformalMap`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.ConformalMap), [`map_derivative`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.map_derivative), [`conformal_grid`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.conformal_grid), [`PolarPreconditioner`](https://danielboigk.github.io/ModularEIT.jl/dev/api/linear_solvers/#ModularEIT.PolarPreconditioner).

## References

1. T. Theodorsen (1931). *Theory of wing sections of arbitrary shape*. NACA Report No. 411. [ntrs.nasa.gov/citations/19930091476](https://ntrs.nasa.gov/citations/19930091476)
2. M. H. Gutknecht (1983). *Numerical Experiments on Solving Theodorsen's Integral Equation for Conformal Maps with the Fast Fourier Transform and Various Nonlinear Iterative Methods*. SIAM J. Sci. Stat. Comput. 4(1), 1–30. [doi:10.1137/0904001](https://doi.org/10.1137/0904001)
3. B. Fornberg (1980). *A Numerical Method for Conformal Mappings*. SIAM J. Sci. Stat. Comput. 1(3), 386–400. [doi:10.1137/0901027](https://doi.org/10.1137/0901027)
4. G. T. Symm (1966). *An integral equation method in conformal mapping*. Numer. Math. 9, 250–258. [doi:10.1007/BF02162088](https://doi.org/10.1007/BF02162088)
5. R. Wegmann (1978). *Ein Iterationsverfahren zur konformen Abbildung*. Numer. Math. 30, 453–466. [doi:10.1007/BF01398511](https://doi.org/10.1007/BF01398511)
6. R. Wegmann (2005). *Methods for numerical conformal mapping*. In: Handbook of Complex Analysis: Geometric Function Theory, Vol. 2, 351–477. [doi:10.1016/S1874-5709(05)80013-7](https://doi.org/10.1016/S1874-5709(05)80013-7)
7. R. Kress (2014). *Linear Integral Equations*, 3rd ed. Springer. [doi:10.1007/978-1-4614-9593-2](https://doi.org/10.1007/978-1-4614-9593-2)
8. L. N. Trefethen (2020). *Numerical conformal mapping with rational functions*. Comput. Methods Funct. Theory 20, 369–387. [doi:10.1007/s40315-020-00325-w](https://doi.org/10.1007/s40315-020-00325-w)
