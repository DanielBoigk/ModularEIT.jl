---
tags: [overview, regularization, machine-learning]
---

Because EIT is severely ill-posed (see [[Stability of the Calderón Problem]]), prior knowledge about the conductivity determines what a reconstruction can show. Priors range from explicit penalties to generative models learned from data.

## Explicit penalties

[[Variational Regularization]] adds a functional $\alpha R(\sigma)$ to the misfit:

- [[Tikhonov Regularization]] favours small or smooth conductivities;
- [[Total Variation]] favours piecewise constant ones with sharp edges ([[Smoothed Total Variation]] makes it differentiable);
- [[Spectral Sobolev Norms on Rectangles]] interpolate between these smoothness classes.

The weight α trades data fit against prior (see [[Choosing the Regularization Parameter]]). Stopping an iteration early also regularises (see [[Implicit Regularization]]). In the Bayesian view, $R$ is a negative log-prior and the reconstruction a posterior mode (see [[Bayesian Inversion]]).

## Priors from data

[[Learned Regularization]] replaces the hand-made $R$ by one fitted to example conductivities (see [[Synthetic Conductivity Data]]):

- **Denoisers as proximal operators.** A trained denoiser can replace the proximal step of a splitting method (see [[Plug-and-Play Priors]], [[Regularization by Denoising]]).
- **Explicit learned energies.** These can be convex by construction (see [[Energy-Based Models]], [[Input Convex Neural Networks]]).
- **Implicit models.** Reconstructions can also be defined as fixed points of learned maps (see [[Deep Equilibrium Models]]).

## Generative priors

[[Diffusion Models]] learn the score of the prior distribution (see [[Score Function]], [[Denoising Score Matching]]). They can be combined with the EIT data term during sampling (see [[Diffusion Posterior Sampling]], [[DiffPIR]], [[Diffusion Models for EIT]]). Instead of a single estimate they produce samples, and with them a measure of uncertainty (see [[Langevin Dynamics]]).

Learned priors can also produce plausible structures that are not in the data (see [[Hallucinations and Uncertainty]]). Symmetries of the problem can be built into the networks (see [[Symmetries of the EIT Problem]]).

## References

1. S. Arridge, P. Maass, O. Öktem, C.-B. Schönlieb (2019). *Solving inverse problems using data-driven models*. Acta Numer. 28, 1–174. [doi:10.1017/S0962492919000059](https://doi.org/10.1017/S0962492919000059)
2. M. Benning, M. Burger (2018). *Modern regularization methods for inverse problems*. Acta Numer. 27, 1–111. [doi:10.1017/S0962492918000016](https://doi.org/10.1017/S0962492918000016)
3. J. Kaipio, E. Somersalo (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
