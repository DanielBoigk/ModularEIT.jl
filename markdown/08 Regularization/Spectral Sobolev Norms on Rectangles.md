---
tags: [numerics, fem, regularization, fft]
aliases: [DCT Sobolev norms, Sobolev gradients on rectangles, Spectral Gaussian random fields]
---

The spectral construction of [[Discrete Fractional Sobolev Norms]] needs the eigenvectors of a pair of matrices $(K, M)$. For functions on a uniform rectangle grid these are known in closed form: they are the [[Discrete Cosine Transform]] modes. Norms of any order $s\in\mathbb R$ then cost $O(n\log n)$, with no eigenvalue computation and no dense matrices. These notes collect the definitions and uses for an implementation.

## Definition

Let $K\Phi = M\Phi\Lambda$ with $\Phi^\top M\Phi = I$ be the generalised eigen decomposition of a stiffness-type matrix $K$ and the mass matrix $M$ of the σ space. For a length scale $\ell>0$ define

$$
A(s) = M\,\Phi\operatorname{diag}\big((1+\ell^2\lambda_k)^s\big)\,\Phi^\top M,\qquad
\Vert\sigma\Vert_{H^s}^2 = \sigma^\top A(s)\,\sigma .
$$

The special cases are $A(0) = M$ and $A(1) = M+\ell^2K$. The inverse is $A(s)^{-1} = \Phi\operatorname{diag}((1+\ell^2\lambda_k)^{-s})\Phi^\top$, and $A(-s) = M A(s)^{-1}M$. On a uniform grid:

- **Piecewise constant σ** (pixels). $M = h_x h_y I$, and $K$ is the two-point-flux jump matrix (the `:jump` penalty). $\Phi$ is the 2D DCT-II, normalised, and $\lambda_{kl} = \big(\tfrac{h_y}{h_x}\mu_k+\tfrac{h_x}{h_y}\mu_l\big)/(h_x h_y)$ with $\mu_k = 2(1-\cos(\pi k/n))$. For small $k$ these approximate the continuous eigenvalues $(\pi k/\ell_x)^2+(\pi l/\ell_y)^2$.
- **Bilinear σ.** $(K, M)$ are the Q1 stiffness and mass matrices, $\Phi$ is the 2D DCT-I, and $\lambda$ is the elementwise ratio of their eigenvalues.

A fractional $s$ gives a dense $A(s)$, which is never formed. Every product with $A(s)$ or $A(s)^{-1}$ is a transform, a diagonal scaling and an inverse transform.

## Uses

**Regulariser.** $R(\sigma) = \tfrac12\Vert\sigma-\sigma_0\Vert_{H^s}^2$ has gradient $A(s)(\sigma-\sigma_0)$ and the constant [[Gauss-Newton Method|Gauss–Newton]] Hessian $A(s)$. For $s = 1$ it is the H¹ [[Tikhonov Regularization]]. Fractional $s\in(0,1)$ penalises oscillations less than H¹ and allows sharper edges.

**Exact proximal operator.** The prox in the $L^2$ metric,

$$
\operatorname{prox}(v) = \arg\min_z\ \tfrac\alpha2\Vert z-\sigma_0\Vert_{H^s}^2+\tfrac\rho2\Vert z-v\Vert_{M}^2
= \Phi\operatorname{diag}\Big(\frac{1}{\alpha(1+\ell^2\lambda_k)^s+\rho}\Big)\Phi^\top\big(\alpha A(s)\sigma_0+\rho Mv\big),
$$

is diagonal in the transform basis. It is exact and costs $O(n\log n)$, so it is the regulariser step of [[ADMM]] for Sobolev priors (see [[Proximal Operator]]).

**Sobolev gradients.** The Riesz representative of a coefficient gradient $g$ in $H^s$ is $A(s)^{-1}g$ (see [[Gradient Representation and the Riesz Map]]). $s = 0$ is the $L^2$ gradient $M^{-1}g$. For $s>0$ the modes are damped by $(1+\ell^2\lambda_k)^{-s}$, which smooths the update on the length scale $\ell$ independently of the mesh. For $s = 1$ this is Neuberger's Sobolev gradient. A first-order method with this Riesz map is preconditioned by a smoothness prior without changing the objective.

**Gaussian random fields.** For white noise $\xi\sim\mathcal N(0,I)$,

$$
\sigma = \Phi\operatorname{diag}\big((1+\ell^2\lambda_k)^{-s/2}\big)\,\xi
$$

has covariance $A(s)^{-1}$. This is a Matérn-type field with correlation length $\ell$ and smoothness controlled by $s$: in $d$ dimensions, $s = \nu+d/2$ corresponds to Matérn smoothness $\nu$, the discrete form of the SPDE $(1-\ell^2\Delta)^{s/2}\sigma = \xi$ with Neumann conditions. It gives [[Synthetic Conductivity Data|synthetic conductivities]] (for example $\exp$ of a field, or thresholded fields for inclusions) and Gaussian priors for [[Bayesian Inversion]] and [[Langevin Dynamics]]. Every sample costs one transform.

**Boundary norms.** The boundary of a rectangle is a closed polygon. With equal spacings $h_x = h_y$ its boundary mass and stiffness matrices are circulant. The FFT then diagonalises them, and the $H^{\pm1/2}$ norms of [[Discrete Fractional Sobolev Norms]] cost $O(n_\Gamma\log n_\Gamma)$.

## Implementation plan

Everything shares the transform plans of [[Fast Solvers on Rectangular Domains]]:

1. `SpectralSobolevRegularizer(disc; s, ℓ, reference)`: value, gradient, Gauss–Newton Hessian as an operator, and `prox`.
2. `SobolevGradient(disc; s, ℓ)`: an `AbstractRieszMap` for gradient descent and L-BFGS.
3. `gaussian_random_field(disc; s, ℓ, rng)` for synthetic data.
4. Tests: $s = 0$ and $s = 1$ agree with the assembled $M$ and $M+\ell^2K$. $A(-s) = M A(s)^{-1} M$. The prox satisfies its optimality condition. The sample covariance of random fields converges to $A(s)^{-1}$.

**In ModularEIT.jl:** [`gaussian_random_field`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.gaussian_random_field).

## References

1. J. W. Neuberger (2010). *Sobolev Gradients and Differential Equations*, 2nd ed. Lecture Notes in Mathematics 1670, Springer. [doi:10.1007/978-3-642-04041-2](https://doi.org/10.1007/978-3-642-04041-2)
2. F. Lindgren, H. Rue, J. Lindström (2011). *An Explicit Link between Gaussian Fields and Gaussian Markov Random Fields: The Stochastic Partial Differential Equation Approach*. J. R. Stat. Soc. B 73(4), 423–498. [doi:10.1111/j.1467-9868.2011.00777.x](https://doi.org/10.1111/j.1467-9868.2011.00777.x)
3. C. R. Dietrich, G. N. Newsam (1997). *Fast and Exact Simulation of Stationary Gaussian Processes through Circulant Embedding of the Covariance Matrix*. SIAM J. Sci. Comput. 18(4), 1088–1107. [doi:10.1137/S1064827592240555](https://doi.org/10.1137/S1064827592240555)
4. G. Strang (1999). *The Discrete Cosine Transform*. SIAM Rev. 41(1), 135–147. [doi:10.1137/S0036144598336745](https://doi.org/10.1137/S0036144598336745)
