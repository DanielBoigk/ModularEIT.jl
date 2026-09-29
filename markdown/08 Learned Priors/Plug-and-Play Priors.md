---
tags: [machine-learning, regularization, splitting]
aliases: [PnP, Plug-and-Play]
---

In a splitting method such as [[ADMM]], the regulariser only enters through its [[Proximal Operator]]:

$$
z^{k+1} = \operatorname{prox}_{\beta\mathcal R/\rho}(x^{k+1}+u^k) = \arg\min_z\ \beta\mathcal R(z)+\tfrac\rho2\|z-(x^{k+1}+u^k)\|^2 .
$$

This is a MAP **denoising** problem: remove Gaussian noise of variance $\beta/\rho$ under the prior $e^{-\mathcal R}$. **Plug-and-Play (PnP)** replaces this step with an arbitrary denoiser $D$ with noise level $\sqrt{\beta/\rho}$, such as BM3D or a trained CNN like DRUNet:

$$
z^{k+1} = D_{\sqrt{\beta/\rho}}\big(x^{k+1}+u^k\big).
$$

The prior is never written down; it is defined implicitly by the denoiser.

**Properties.**

- *Modular*: the physics ($x$-step, with an EIT data prox) and the prior ($z$-step) are completely decoupled. One trained denoiser works for any forward operator (operator-agnostic; see [[Learned Regularization]]).
- *Convergence*: a general denoiser is not the prox of any function, so convergence needs assumptions. Examples are non-expansive or averaged denoisers, or MMSE denoisers, for which nonconvex PnP-ADMM convergence has been shown (Park et al. 2023). Without such conditions iterates can oscillate. Decreasing the noise level of the denoiser over the iterations helps in practice.
- Variants exist for half-quadratic splitting (DPIR; Zhang et al. 2021) and proximal gradient methods.

The diffusion-model analogue is [[DiffPIR]]. A related explicit construction is [[Regularization by Denoising]].

**In ModularEIT.jl:** [`ProximalMap`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.ProximalMap), [`ADMM`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.ADMM).

## References

1. S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg (2013). *Plug-and-Play priors for model based reconstruction*. IEEE GlobalSIP 2013, 945–948. [doi:10.1109/GlobalSIP.2013.6737048](https://doi.org/10.1109/GlobalSIP.2013.6737048)
2. K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, R. Timofte (2021). *Plug-and-Play Image Restoration with Deep Denoiser Prior*. IEEE TPAMI 44(10). [arXiv:2008.13751](https://arxiv.org/abs/2008.13751)
3. C. Park, S. Shoushtari, W. Gan, U. S. Kamilov (2023). *Convergence of Nonconvex PnP-ADMM with MMSE Denoisers*. IEEE CAMSAP 2023, 511–515. [doi:10.1109/CAMSAP58249.2023.10403463](https://doi.org/10.1109/CAMSAP58249.2023.10403463)
4. E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, W. Yin (2019). *Plug-and-Play Methods Provably Converge with Properly Trained Denoisers*. ICML 2019. [arXiv:1905.05406](https://arxiv.org/abs/1905.05406)
