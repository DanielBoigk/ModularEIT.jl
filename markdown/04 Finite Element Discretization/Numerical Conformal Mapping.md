---
tags: [numerics, geometry, fft]
aliases: [Theodorsen's method, Riemann map, Conformally mapped meshes]
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

**Convergence.** The iteration contracts if the boundary is *$\varepsilon$-nearly circular*, $|\mathrm d\log\rho/\mathrm d\vartheta|\le\varepsilon<1$; the error decreases like $\varepsilon^k$. For an ellipse with semi-axes $a>b$, $\varepsilon = (a^2-b^2)/(2ab)$, so aspect ratios up to about $2.4$ are admissible. Thorax- and head-shaped cross-sections are well within the range. Under-relaxation widens it somewhat. Domains far from circular, or not star-shaped, need other methods: Wegmann's Newton-type method, Fornberg's method, Schwarz–Christoffel maps for polygons, or rational approximation.

**Accuracy.** The discrete map matches the boundary exactly at the $N$ nodes. Between the nodes the error is the trigonometric interpolation error of $\log\rho\circ\vartheta$. It is spectrally small for smooth boundaries. At corners the boundary correspondence has singular derivatives, and convergence in $N$ is algebraic.

**Crowding.** For elongated domains $|\Phi'|$ becomes exponentially small on parts of the boundary. Mapped grids then concentrate nodes where the domain is wide, and a rectangle is the better model domain.

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
4. L. N. Trefethen (2020). *Numerical conformal mapping with rational functions*. Comput. Methods Funct. Theory 20, 369–387. [doi:10.1007/s40315-020-00325-w](https://doi.org/10.1007/s40315-020-00325-w)
