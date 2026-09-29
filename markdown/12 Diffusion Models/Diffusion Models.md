---
tags: [machine-learning, generative, overview]
aliases: [Score-based diffusion, Diffusion model]
---

**Diffusion models** learn a data distribution $p_{\text{data}}$ by learning to *reverse* a process that gradually turns data into Gaussian noise.

1. **Forward (noising) process.** Corrupt data with increasing amounts of Gaussian noise until only noise remains (see [[DDPM Forward Process]], [[Noise Schedule]] and [[Variance-Preserving SDE]]).
2. **Learning.** Train a network to predict the added noise, or equivalently the [[Score Function]] $\nabla_x\log p_t(x)$ of the noisy marginals, at every noise level (see [[Denoising Score Matching]]).
3. **Generation.** Start from pure noise and integrate the [[Reverse-Time SDE]] or the [[Probability Flow ODE]] with the learned score (see [[Euler-Maruyama Method]] and [[DDPM Ancestral Sampling]]).

```tikz
\begin{document}
\begin{tikzpicture}[scale=1]
  \foreach \i/\l in {0/{$x_0$},2.2/{$x_{t}$},4.4/{$x_{t'}$},6.6/{$x_T$}} {
    \draw[thick, rounded corners] (\i,0) rectangle (\i+1.2,1);
    \node at (\i+0.6,0.5) {\l};
  }
  \draw[->, thick] (1.3,0.75) -- (2.1,0.75);
  \draw[->, thick] (3.5,0.75) -- (4.3,0.75);
  \draw[->, thick] (5.7,0.75) -- (6.5,0.75);
  \draw[<-, thick, dashed] (1.3,0.25) -- (2.1,0.25);
  \draw[<-, thick, dashed] (3.5,0.25) -- (4.3,0.25);
  \draw[<-, thick, dashed] (5.7,0.25) -- (6.5,0.25);
  \node at (3.9,1.4) {forward: add noise};
  \node at (3.9,-0.4) {reverse: denoise with learned score};
  \node at (0.6,-1) {data};
  \node at (7.2,-1) {$\mathcal N(0,I)$};
\end{tikzpicture}
\end{document}
```

**As priors for inverse problems.** A trained diffusion model is an operator-agnostic prior (see [[Learned Regularization]]). It is combined with the EIT forward model in three main ways:

- guidance of the reverse process with data gradients ([[Diffusion Posterior Sampling]]);
- alternating denoising and data-consistency prox steps ([[DiffPIR]]);
- a variational regulariser, independent of the sampling schedule ([[RED-Diff]] and [[Diffusion Proximal Operator]]).

All three rely on [[Tweedie's Formula]] to estimate the clean image from a noisy one. For EIT-specific work see [[Diffusion Models for EIT]].

## References

1. J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
2. J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 33. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
3. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
