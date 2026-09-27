---
tags: [regularization, objective]
aliases: [Data misfit, Loss function, Metric]
---

The data fidelity term $d(\mathcal F(\sigma), y)$ in a [[Variational Regularization|variational reconstruction]] measures how far the predicted boundary data are from the measured data.

**Output least squares (boundary $L^2$).** For Neumann data $g_i$ and measured voltages $f_i$:

$$
J(\sigma) = \sum_{i=1}^N \tfrac12\,\big\| u_i(\sigma)|_{\partial\Omega} - f_i\big\|_{L^2(\partial\Omega)}^2 ,
$$

where $u_i(\sigma)$ solves the [[Neumann Problem]]. This is the most common choice. Discretely, $\|v\|^2_{L^2(\partial\Omega)} = v^\top M_\Gamma v$ with the [[Boundary Mass and Stiffness Matrices|boundary mass matrix]]. The plain Euclidean norm of nodal values only approximates it on a uniform boundary mesh.

**Noise-weighted.** For Gaussian noise with covariance $\Gamma$, use $\tfrac12\|\cdot\|_{\Gamma^{-1}}^2$ (see [[Bayesian Inversion]]).

**Sobolev norms.** Measuring the misfit in $H^{1/2}(\partial\Omega)$ or $H^{-1/2}(\partial\Omega)$ matches the natural function spaces of the [[Neumann-to-Dirichlet Map]] and weights frequencies differently (see [[Discrete Fractional Sobolev Norms]]).

**Energy (Kohn–Vogelius).** An interior mismatch between Dirichlet and Neumann solutions (see [[Kohn-Vogelius Functional]]). It is equivalent to a data fit in the energy norm.

**Optimal transport.** Wasserstein distances between boundary data, treated as densities after normalisation, give smoother landscapes with fewer spurious local minima (Bao & Zhang 2022).

**Operator-level.** When a whole discrete operator $\Lambda_{\text{meas}}$ is available, one can compare operators, for example in Frobenius or operator norm, possibly after a spectral truncation (see [[Truncated SVD Regularization]]).

Any differentiable $d$ fits the [[Adjoint State Method]]. Only the right-hand side of the adjoint equation, $\partial_u d$, changes.

## References

1. L. Borcea (2002). *Electrical impedance tomography*. Inverse Problems 18(6), R99–R136. [doi:10.1088/0266-5611/18/6/201](https://doi.org/10.1088/0266-5611/18/6/201)
2. G. Bao, Y. Zhang (2022). *Optimal Transportation for Electrical Impedance Tomography*. [arXiv:2210.16082](https://arxiv.org/abs/2210.16082)
3. J. Kaipio, E. Somersalo (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
