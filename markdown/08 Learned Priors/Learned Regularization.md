---
tags: [machine-learning, regularization]
aliases: [Data-driven regularization, Learned prior]
---

Hand-crafted regularisers ([[Tikhonov Regularization]], [[Total Variation]]) encode generic assumptions such as smoothness or piecewise constancy. **Learned regularisation** instead estimates the prior from examples of plausible conductivities $\sigma\sim p_{\text{data}}$, for example human tissue or images (see [[Synthetic Conductivity Data]]).

**Two training regimes.**

1. **Operator-agnostic.** Train on the conductivity distribution alone, with no forward solves. Examples are a denoiser, a score model or an energy. The same prior works with any forward operator, electrode setup or noise level. Examples: [[Plug-and-Play Priors]], [[Regularization by Denoising]], [[Diffusion Models]].
2. **Operator-aware.** Train with the forward model in the loop, for example on intermediate iterates of actual reconstructions, or by unrolling an optimiser. The prior then learns the specific artefacts of EIT, but it is tied to one setup and training is expensive.

**Families of learned regularisers.**

- *Explicit* functionals $\mathcal R_\theta(\sigma)$: adversarial regularisers, [[Input Convex Neural Networks|convex]] or weakly convex networks with guarantees (Mukherjee et al.; Goujon et al.), and [[Energy-Based Models]].
- *Implicit* via denoisers: plug-and-play, RED.
- *Generative*: autoencoder or [[Manifold Hypothesis|manifold]] penalties, normalising flows, and diffusion or score models ([[Diffusion Posterior Sampling]], [[DiffPIR]], [[RED-Diff]]).
- *Architectural*: iterative networks such as [[Neural ODEs]] and [[Deep Equilibrium Models]] that map a degraded image to a clean one.

**Risks.** Learned priors can **hallucinate** structure that the data do not support, especially in the weakly determined interior of EIT (see [[Hallucinations and Uncertainty]]). Convex or provably stable regularisers trade expressiveness for guarantees.

## References

1. S. Arridge, P. Maass, O. Öktem, C.-B. Schönlieb (2019). *Solving inverse problems using data-driven models*. Acta Numerica 28, 1–174. [doi:10.1017/S0962492919000059](https://doi.org/10.1017/S0962492919000059)
2. M. Haltmeier, L. V. Nguyen (2020). *Regularization of Inverse Problems by Neural Networks*. [arXiv:2006.03972](https://arxiv.org/abs/2006.03972)
3. S. Lunz, O. Öktem, C.-B. Schönlieb (2018). *Adversarial Regularizers in Inverse Problems*. NeurIPS 31. [arXiv:1805.11572](https://arxiv.org/abs/1805.11572)
4. S. Mukherjee, S. Dittmer, Z. Shumaylov, S. Lunz, O. Öktem, C.-B. Schönlieb (2021). *Learned convex regularizers for inverse problems*. [arXiv:2008.02839](https://arxiv.org/abs/2008.02839)
5. A. Goujon, S. Neumayer, P. Bohra, S. Ducotterd, M. Unser (2023). *A Neural-Network-Based Convex Regularizer for Inverse Problems*. IEEE Trans. Comput. Imaging 9, 781–795. [arXiv:2211.12461](https://arxiv.org/abs/2211.12461)
