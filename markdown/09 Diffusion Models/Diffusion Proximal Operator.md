---
tags: [diffusion, splitting]
aliases: [Diffusion prox, RED-Diff prox]
---

To use a diffusion prior inside [[ADMM]] (see [[Nested ADMM Reconstruction]]), define the diffusion regulariser

$$
E_{\text{diff}}(x) = \mathbb E_{t,\varepsilon}\Big[\omega(t)\,\big\|\varepsilon_\theta\big(\sqrt{\bar\alpha_t}\,x+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ t\big)-\varepsilon\big\|^2\Big]
$$

(the [[RED-Diff]] regulariser) and approximate its [[Proximal Operator]]:

$$
\operatorname{prox}_{E_{\text{diff}}/\rho}(v) = \arg\min_x\ E_{\text{diff}}(x)+\frac\rho2\|x-v\|^2 .
$$

**Monte Carlo estimate** with $n$ samples, evaluated in parallel as one batch:

```text
function error_diff(x; n)
    for i = 1…n (batched):
        t_i ~ schedule,  ε_i ~ N(0, I)
        x_i ← sqrt(ᾱ_{t_i}) x + sqrt(1−ᾱ_{t_i}) ε_i
        r_i ← ε_θ(x_i, t_i) − ε_i
    E  ← (1/n) Σ ω(t_i) ‖r_i‖²
    ∇E ← (1/n) Σ λ(t_i) r_i            (stop-gradient through ε_θ)
    return E, ∇E
```

**Prox by stochastic optimisation.** Starting at $x = v$, iterate $x\leftarrow x-\eta\big(\nabla E+\rho\,P(x-v)\big)$, or use Adam, for a fixed budget. Here $P$ is optional: a weighting or projection that chooses *where* deviations from $v$ are penalised. For example, a boundary-weighted $P$ keeps the well-determined region near the electrodes close to the data-consistent iterate and lets the prior act mainly in the interior.

**Remarks.**

- Because the gradient is stochastic and not the exact gradient of $E$ (the network Jacobian is dropped), this is an *approximate* prox. Convergence analyses of related proximal stochastic denoising schemes exist (Renaud et al. 2024).
- Unlike [[DiffPIR]] and [[Diffusion Posterior Sampling]], it needs no knowledge of the current diffusion time. It can therefore run asynchronously alongside the physics solver.
- The same construction can be used as a post-processing step on any reconstruction.

## References

1. M. Mardani, J. Song, J. Kautz, A. Vahdat (2024). *A Variational Perspective on Solving Inverse Problems with Diffusion Models*. ICLR 2024. [arXiv:2305.04391](https://arxiv.org/abs/2305.04391)
2. M. Renaud, J. Hermant, N. Papadakis (2024). *Convergence Analysis of a Proximal Stochastic Denoising Regularization Algorithm*. [arXiv:2412.08262](https://arxiv.org/abs/2412.08262)
3. S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein (2011). *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. Found. Trends Mach. Learn. 3(1), 1–122. [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)
