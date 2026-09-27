---
tags: [reconstruction, optimization]
---

A modular EIT reconstruction can be organised in three nested layers. Each layer can be exchanged independently.

**1. Objective and gradient (innermost).**

- `obj(σ)`: solve the [[State Equation]] for every current pattern and return the residuals and the misfit.
- `grad(σ)`: reuse the state solutions, solve the [[Adjoint Equation]] per pattern, assemble $-\nabla u_i\cdot\nabla\lambda_i$, and combine the per-pattern gradients. Combination can be a plain sum, or a single [[Gauss-Newton Method|Gauss–Newton]] / [[Levenberg-Marquardt Method|Levenberg–Marquardt]] direction.

**2. Data-fidelity proximal operator (middle).**

$$
\operatorname{prox}_{\Phi/\rho}(y) = \arg\min_\sigma\ \Phi(\sigma)+\frac\rho2\|\sigma-y\|^2 ,
$$

approximated by a few iterations of [[L-BFGS-B]] (with gradient $\nabla\Phi+\rho(\sigma-y)$ and [[Box Constraints on Conductivity|bounds]]) or of Gauss–Newton with a [[Line Search]], warm-started at $y$.

**3. ADMM (outermost).**

```text
x, z, u ← σ0, σ0, 0
repeat
    x ← prox_{Φ/ρ}(z − u)          data consistency
    z ← prox_{βR/ρ}(x + u)         regulariser: TV, Tikhonov, denoiser, ...
    u ← u + x − z                   scaled dual update
until ‖x − z‖ and ρ‖z − z_old‖ are small
return z
```

(see [[ADMM]]).

**Why split?** The expensive, nonconvex physics and the cheap or non-smooth prior are handled by the tools best suited to each. The regulariser can be swapped, for example for [[Plug-and-Play Priors|a learned denoiser]], without touching the PDE code. The inner proximal problem only needs to be solved approximately; inexact ADMM tolerates this as long as the errors are summable.

## References

1. S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein (2011). *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. Found. Trends Mach. Learn. 3(1), 1–122. [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)
2. S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg (2013). *Plug-and-Play priors for model based reconstruction*. IEEE GlobalSIP 2013, 945–948. [doi:10.1109/GlobalSIP.2013.6737048](https://doi.org/10.1109/GlobalSIP.2013.6737048)
3. J. Eckstein, D. P. Bertsekas (1992). *On the Douglas–Rachford splitting method and the proximal point algorithm for maximal monotone operators*. Math. Program. 55, 293–318. [doi:10.1007/BF01581204](https://doi.org/10.1007/BF01581204)
