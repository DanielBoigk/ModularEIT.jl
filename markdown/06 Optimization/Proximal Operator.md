---
tags: [optimization, convex-analysis]
aliases: [Prox, Proximal map]
---

For a function $R:\mathbb R^n\to\mathbb R\cup\{+\infty\}$ and a parameter $\rho>0$, the **proximal operator** is

$$
\operatorname{prox}_{R/\rho}(v) = \arg\min_z\ R(z)+\frac\rho2\|z-v\|_2^2 .
$$

It moves $v$ towards smaller values of $R$ while staying close to $v$.

**Properties** (for convex, lower semicontinuous $R$):

- the minimiser exists and is unique; the map is firmly non-expansive (1-Lipschitz);
- fixed points of $\operatorname{prox}_{R/\rho}$ are exactly the minimisers of $R$;
- it is well defined for **non-smooth** $R$, such as [[Total Variation]], the $\ell_1$ norm, or indicator functions of constraint sets. For an indicator $\iota_C$ it is the projection onto $C$;
- for differentiable $R$: $z = v-\rho^{-1}\nabla R(z)$, an *implicit* gradient step.

**Examples.**

| $R(z)$ | $\operatorname{prox}_{R/\rho}(v)$ |
|:--|:--|
| $\tfrac\beta2 z^\top Kz$ | $(\beta K+\rho I)^{-1}\rho v$ (see [[Tikhonov Regularization]]) |
| $\beta\Vert z\Vert_1$ | soft thresholding, $\operatorname{sign}(v)\max(\vert v\vert-\beta/\rho,0)$ |
| $\iota_{[a,b]^n}$ | clipping to $[a,b]$ |
| $\beta\operatorname{TV}(z)$ | ROF denoising (see [[Total Variation]]) |

**Prox of the data term.** For the nonconvex EIT misfit $\Phi_{\text{data}}$, $\operatorname{prox}_{\Phi_{\text{data}}/\rho}(v) = \arg\min_\sigma\Phi_{\text{data}}(\sigma)+\frac\rho2\|\sigma-v\|^2$ has no closed form. It is computed approximately by a few iterations of [[L-BFGS-B]] or Gauss–Newton, with the extra gradient term $\rho(\sigma-v)$.

**Denoisers as proximal operators.** A denoiser $D$ maps a noisy image to a clean one, just as a prox maps $v$ to a nearby point with small $R$. Replacing $\operatorname{prox}_R$ by a learned denoiser is the idea behind [[Plug-and-Play Priors]].

**In ModularEIT.jl:** [`prox!`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.prox!), [`ProximalMap`](https://danielboigk.github.io/ModularEIT.jl/dev/api/optimization/#ModularEIT.ProximalMap).

## References

1. N. Parikh, S. Boyd (2014). *Proximal Algorithms*. Found. Trends Optim. 1(3), 127–239. [doi:10.1561/2400000003](https://doi.org/10.1561/2400000003)
2. H. H. Bauschke, P. L. Combettes (2017). *Convex Analysis and Monotone Operator Theory in Hilbert Spaces*, 2nd ed. Springer. [doi:10.1007/978-3-319-48311-5](https://doi.org/10.1007/978-3-319-48311-5)
