---
title: Diffusion Models
tags: [overview, machine-learning, diffusion]
---

Diffusion models learn a prior distribution by learning to reverse a gradual noising process. As priors for inverse problems they are combined with the data term during sampling.

**Reading order.**

1. The model: [[Diffusion Models]], [[DDPM Forward Process]], [[Noise Schedule]], [[DDPM Ancestral Sampling]], [[Sinusoidal Time Embedding]].
2. The continuous view: [[Continuous Limit of the DDPM Chain]], [[Variance-Preserving SDE]], [[Reverse-Time SDE]], [[Probability Flow ODE]], [[Euler-Maruyama Method]].
3. Scores: [[Score Function]], [[Denoising Score Matching]], [[Tweedie's Formula]].
4. Inverse problems: [[Diffusion Posterior Sampling]], [[DiffPIR]], [[RED-Diff]], [[Diffusion Proximal Operator]], and [[Diffusion Models for EIT]].

**Related.** Other learned priors: [[11 Learned Priors/index|Learned Priors]]; the broader picture: [[Classical and Learned Priors]].

## References

1. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
3. H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye (2023). *Diffusion Posterior Sampling for General Noisy Inverse Problems*. ICLR 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
