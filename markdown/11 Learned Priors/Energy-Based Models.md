---
tags: [machine-learning, generative]
aliases: [EBM]
---

An **energy-based model** represents a probability density through a learned scalar energy $E_\theta$:

$$
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta},\qquad Z_\theta = \int e^{-E_\theta(x)}\,\mathrm dx .
$$

Low energy means high probability. The normalising constant $Z_\theta$ is intractable, but many tasks do not need it:

- the **score** $\nabla_x\log p_\theta = -\nabla_xE_\theta$ is free of $Z_\theta$ (see [[Score Function]]);
- **sampling** works with [[Langevin Dynamics]];
- as a **regulariser**, $\mathcal R(\sigma) = E_\theta(\sigma)$ plugs directly into a [[Variational Regularization|variational reconstruction]] (MAP estimate), and its gradient comes from autodiff.

**Training.**

- *Maximum likelihood / contrastive divergence:* $\nabla_\theta\,\mathbb E[-\log p_\theta] = \mathbb E_{\text{data}}[\nabla_\theta E_\theta]-\mathbb E_{p_\theta}[\nabla_\theta E_\theta]$. The negative phase needs samples from the model, usually short-run Langevin chains.
- *Score matching / denoising score matching:* fit $-\nabla E_\theta$ to the score of noise-perturbed data (see [[Denoising Score Matching]]). This avoids sampling.

**Why interesting for EIT.** An explicit energy gives an objective that can be evaluated, a gradient, and a well-defined [[Proximal Operator]]. These plug into the same optimisers as the physics. Diffusion models only provide scores, not energies.

## References

1. Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, F. J. Huang (2006). *A Tutorial on Energy-Based Learning*. In: Predicting Structured Data, MIT Press. [yann.lecun.com/exdb/publis/pdf/lecun-06.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
2. Y. Du, I. Mordatch (2019). *Implicit Generation and Generalization in Energy-Based Models*. NeurIPS 32. [arXiv:1903.08689](https://arxiv.org/abs/1903.08689)
3. Y. Song, D. P. Kingma (2021). *How to Train Your Energy-Based Models*. [arXiv:2101.03288](https://arxiv.org/abs/2101.03288)
