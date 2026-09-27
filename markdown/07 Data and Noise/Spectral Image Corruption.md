---
tags: [data, machine-learning]
aliases: [Synthetic degradation, DCT corruption]
---

To train a denoiser or deblurrer that should later clean EIT reconstructions, one needs pairs (clean, degraded). Real EIT reconstructions are expensive to produce, so a cheap **synthetic degradation** is used that imitates their typical artefacts: loss of fine detail, spatially varying blur, and structured noise.

**Discrete cosine transform (DCT).** For an $n\times m$ image $x$, the 2D DCT-II $C = \mathrm{DCT}(x)$ expresses $x$ in cosine modes $\cos\big(\tfrac{\pi k(2i+1)}{2n}\big)\cos\big(\tfrac{\pi\ell(2j+1)}{2m}\big)$. These are the eigenfunctions of the discrete Laplacian with reflecting (Neumann) boundaries. Frequency $(k,\ell)$ has squared magnitude $|\omega|^2 = k^2+\ell^2$.

**Iterated corruption.** Repeat $T$ times:

1. add spatial white noise: $x\leftarrow x+\sigma_{\text{sp}}\,\xi$;
2. transform: $C\leftarrow\mathrm{DCT}(x)$;
3. add frequency-dependent noise: $C_{k\ell}\leftarrow C_{k\ell}+\sigma_{\text{fr}}\,(1+|\omega|^2)\,\xi_{k\ell}$;
4. damp high frequencies: $C_{k\ell}\leftarrow e^{-\lambda|\omega|^{2s}}C_{k\ell}$;
5. transform back: $x\leftarrow\mathrm{DCT}^{-1}(C)$.

Step 4 alone is the solution operator of the heat equation (for $s=1$), a linear Gaussian-type blur. Combined with repeated noise injection, the result is a random, nonstationary degradation whose strength is controlled by $T$, $\lambda$, $s$ and the noise levels.

**Caveat.** This is only a proxy. EIT artefacts depend on depth (the resolution loss grows towards the centre; see [[Decay of Boundary Measurements]]) and on the specific algorithm. Training on actual reconstructions, or learning inside the reconstruction loop, captures the operator-specific degradation better (see [[Learned Regularization]]).

## References

1. N. Ahmed, T. Natarajan, K. R. Rao (1974). *Discrete Cosine Transform*. IEEE Trans. Comput. C-23(1), 90–93. [doi:10.1109/T-C.1974.223784](https://doi.org/10.1109/T-C.1974.223784)
2. G. Strang (1999). *The Discrete Cosine Transform*. SIAM Review 41(1), 135–147. [doi:10.1137/S0036144598336745](https://doi.org/10.1137/S0036144598336745)
