---
tags: [optimization]
---

Common rules for terminating a reconstruction:

- **Discrepancy principle:** stop when $\|\mathcal F(\sigma_k)-y^\delta\|\le\tau\delta$ ($\tau\gtrsim1$, $\delta$ the noise level). Iterating further fits noise, so for ill-posed problems this is the principled rule (see [[Choosing the Regularization Parameter]] and [[Implicit Regularization]]).
- **Stationarity:** $\|\nabla\Phi(\sigma_k)\|\le\varepsilon_{\text{abs}}+\varepsilon_{\text{rel}}\|\nabla\Phi(\sigma_0)\|$. For bound constraints use the projected gradient $\|P_\Sigma(\sigma-\nabla\Phi)-\sigma\|$ (see [[KKT Conditions]]).
- **Stagnation:** relative change of the objective $|\Phi_{k}-\Phi_{k+1}|/|\Phi_k|$ or of the iterate $\|\sigma_{k+1}-\sigma_k\|/\|\sigma_k\|$ below a tolerance.
- **Stagnation at the noise level:** a step that decreases the whitened misfit by less than a few standard deviations $\sqrt{m/2}$ of its noise-only value is rejected, and the previous iterate returned. It needs no target value, so it also works when the noise model is only estimated (see [[Noise-Level Stagnation Test]]).
- **Splitting methods:** primal and dual residuals of [[ADMM]] below tolerance.
- **Budget:** a maximal number of iterations or forward solves.

Tolerances of the inner linear solves should be tied to the outer progress. Solving to $10^{-12}$ when the gradient is still large wastes effort, while solving too loosely produces gradient noise that stalls line searches.

**In ModularEIT.jl:** [`minimize`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.minimize), [`OptimizationState`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.OptimizationState), [`discrepancy_target`](https://danielboigk.github.io/ModularEIT.jl/dev/api/data/#ModularEIT.discrepancy_target).

## References

1. J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed. Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
2. H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
3. M. Hanke (1997). *A regularizing Levenberg–Marquardt scheme, with applications to inverse groundwater filtration problems*. Inverse Problems 13(1), 79–95. [doi:10.1088/0266-5611/13/1/007](https://doi.org/10.1088/0266-5611/13/1/007)
