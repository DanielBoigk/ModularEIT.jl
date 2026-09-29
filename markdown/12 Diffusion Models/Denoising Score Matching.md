---
tags: [diffusion, training]
aliases: [DSM, Epsilon prediction]
---

The marginal score $\nabla\log p_t$ is unknown, but it can be learned from samples without knowing it.

**Denoising score matching (Vincent 2011).** For any noise kernel $q(x_t\mid x_0)$,

$$
\mathbb E_{x_t}\big\|s_\theta(x_t)-\nabla\log p_t(x_t)\big\|^2 = \mathbb E_{x_0,x_t}\big\|s_\theta(x_t)-\nabla_{x_t}\log q(x_t\mid x_0)\big\|^2+\text{const}.
$$

Regressing onto the *conditional* score, which is known in closed form, gives the *marginal* score at the optimum.

**$\varepsilon$-prediction loss (DDPM).** Inserting the Gaussian kernel of the [[DDPM Forward Process]] and parametrising $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ gives the simple training objective

$$
\mathcal L(\theta) = \mathbb E_{t,\,x_0,\,\varepsilon}\Big[w(t)\,\big\|\varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ t\big)-\varepsilon\big\|^2\Big],
$$

with $t\sim\mathcal U[0,T]$, $x_0\sim p_{\text{data}}$, $\varepsilon\sim\mathcal N(0,I)$ and weight $w(t) = 1$ in DDPM. The network sees the noisy image and the time (via a [[Sinusoidal Time Embedding]]) and predicts the noise. Its optimum is $\varepsilon_\theta(x_t,t) = \mathbb E[\varepsilon\mid x_t]$.

**Training loop.** Sample a batch of images, times and noise; form $x_t$; take a gradient step on the squared error. No simulation of the SDE and no ODE solver are needed, which makes training cheap compared with [[Neural ODEs]]. Other parametrisations predict $x_0$ or $v = \sqrt{\bar\alpha}\,\varepsilon-\sqrt{1-\bar\alpha}\,x_0$; they differ only in the implied weighting $w(t)$.

## References

1. P. Vincent (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation 23(7), 1661–1674. [doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
2. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 33. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
3. T. Salimans, J. Ho (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR 2022. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
