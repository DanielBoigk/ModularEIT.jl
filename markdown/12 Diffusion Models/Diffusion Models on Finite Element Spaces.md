---
tags: [diffusion, fem, function-space]
aliases: [Function-space diffusion, White noise on meshes, Mesh-consistent diffusion]
---

A diffusion model adds Gaussian noise to its samples (see [[DDPM Forward Process]]). On a uniform pixel grid "independent noise per pixel" is unambiguous. On a finite element mesh with elements of different size it is not. Independent noise per node puts the same variance into a tiny element at the boundary as into a large one in the interior. The noised *function* then depends on the mesh, and a network trained on one mesh does not transfer to another.

## White noise, discretised

Gaussian white noise $W$ on $L^2(\Omega)$ has $\mathbb E[\langle W,\varphi\rangle\langle W,\psi\rangle] = \langle\varphi,\psi\rangle_{L^2}$. Its projection onto a finite element space with basis $\varphi_a$ and [[Mass Matrix]] $M$ has coefficients

$$
\xi\sim\mathcal N(0, M^{-1}),\qquad\text{e.g.}\quad \xi = L^{-\top}z,\ \ M = LL^\top,\ \ z\sim\mathcal N(0,I),
$$

or, with the lumped mass matrix, $\xi_a = z_a/\sqrt{m_a}$. Small elements get *more* nodal variance, large elements less, and the noise has the same function-space meaning on every mesh. On a uniform pixel grid $M$ is a multiple of the identity, which is the standard diffusion model.

The same metric enters everywhere a norm appears: the denoising loss $\lVert\hat\varepsilon-\varepsilon\rVert^2_M$, [[Tweedie's Formula]], and the proximal steps of the data consistency (see [[DiffPIR]]).

## Beyond white noise

White noise is not a function: the variance per node grows without bound under refinement. Diffusion models *in function space* therefore use trace-class noise, e.g. a Gaussian random field with covariance $(\kappa^2-\Delta)^{-s}$ for $s > d/2$ (Kerrigan et al. 2023; Lim et al. 2023), discretised with the stiffness and mass matrices. Samples, scores and networks then converge as the mesh is refined, and a model trained at one resolution can be evaluated at another. Such a covariance is also a natural smoothness prior (see [[Tikhonov Regularization]]).

## Pixels as a detour

A [[Parametrizations of the Conductivity|pixel parametrisation]] side-steps the issue: the diffusion model lives on uniform pixels, the forward model on the adapted mesh, and the map between them carries the mesh-dependence. That is the simplest choice when a fixed domain fits a rectangle. Networks that work on the mesh itself (see [[Graph Convolutions on Finite Element Meshes]]) need the mass-weighted noise above.

## References

1. G. Kerrigan, J. Ley, P. Smyth (2023). *Diffusion Generative Models in Infinite Dimensions*. AISTATS 2023. [arXiv:2212.00886](https://arxiv.org/abs/2212.00886)
2. J. H. Lim, N. B. Kovachki, R. Baptista, C. Beckham, K. Azizzadenesheli, J. Kossaifi, V. Voleti, J. Song, K. Kreis, J. Kautz, C. Pal, A. Vahdat, A. Anandkumar (2023). *Score-based Diffusion Models in Function Space*. [arXiv:2302.07400](https://arxiv.org/abs/2302.07400)
