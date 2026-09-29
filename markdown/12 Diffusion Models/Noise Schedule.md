---
tags: [diffusion]
aliases: [Beta schedule, Linear schedule]
---

The **noise schedule** fixes how quickly the [[DDPM Forward Process]] destroys the signal.

**Linear schedule (continuous time).** On $t\in[0,T]$:

$$
\beta(t) = \beta_{\min}+\frac{\beta_{\max}-\beta_{\min}}{T}\,t .
$$

In the continuous limit the cumulative signal factor is (see [[Continuous Limit of the DDPM Chain]])

$$
\bar\alpha(t) = \exp\Big(-\int_0^t\beta(s)\,\mathrm ds\Big) = \exp\Big(-\beta_{\min}t-\frac{\beta_{\max}-\beta_{\min}}{2T}\,t^2\Big).
$$

Equivalently, $\beta(t) = -\frac{\mathrm d}{\mathrm dt}\log\bar\alpha(t)$. Song et al. use $T = 1$, $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$. The discrete DDPM uses $\beta_t$ linear from $10^{-4}$ to $0.02$ over $1000$ steps, which is the same schedule under $\beta_i = \beta(t_i)\Delta t$.

**Signal-to-noise ratio.** $\mathrm{SNR}(t) = \bar\alpha(t)/(1-\bar\alpha(t))$ decreases monotonically from $\infty$ to about $0$. Weighting functions in training and in [[RED-Diff]] are often expressed through it.

**Other schedules.** The cosine schedule (Nichol & Dhariwal 2021) sets $\bar\alpha_t = \cos^2\!\big(\frac{t/T+s}{1+s}\frac\pi2\big)$ (normalised to $\bar\alpha_0 = 1$) and destroys information more evenly on small images. EDM (Karras et al. 2022) parametrises directly by the noise level $\sigma$.

## References

1. Y. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
2. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 33. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
3. A. Nichol, P. Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
4. T. Karras, M. Aittala, T. Aila, S. Laine (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS 35. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
