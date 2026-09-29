---
tags: [machine-learning, architecture, diffusion]
aliases: [Time embedding, Positional encoding]
---

A diffusion denoiser $\varepsilon_\theta(x,t)$ must know the noise level. A scalar $t$ is fed in through a **sinusoidal embedding** into a $d$-dimensional vector:

$$
\operatorname{emb}(t)_{2k} = \sin\big(t\,\omega_k\big),\qquad \operatorname{emb}(t)_{2k+1} = \cos\big(t\,\omega_k\big),\qquad \omega_k = 10000^{-2k/d},
$$

for $k = 0,\dots,d/2-1$. The frequencies are geometrically spaced, so the network can resolve both coarse and fine differences in $t$. The embedding is typically passed through a small MLP and *added* to the feature maps of every residual block of the [[U-Net]].

For continuous $t\in[0,1]$, $t$ is scaled, for example by 1000, before embedding. Random Fourier features ($\omega_k$ drawn from a Gaussian) are an alternative.

## References

1. A. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS 30. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 33. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
3. M. Tancik et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 33. [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)
