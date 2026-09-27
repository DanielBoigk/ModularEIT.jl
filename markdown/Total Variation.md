---
tags: [regularization]
aliases: [TV]
---

**Total variation** regularization favours piecewise-constant conductivities, which
makes it well suited to [[Electrical Impedance Tomography]] of organs or inclusions
with sharp boundaries:

$$
R(\sigma) = \alpha \int_\Omega \lvert \nabla \sigma \rvert \,\mathrm{d}x
\approx \alpha \sum_{k} \sqrt{(\sigma_{k+1} - \sigma_k)^2 + \varepsilon^2}.
$$

The smoothing parameter $\varepsilon > 0$ makes the functional differentiable so that
the same gradient descent as for [[Tikhonov Regularization]] can be used.

## Comparison

| Regularizer                | Edges     | Cost    |
|:---------------------------|:----------|:--------|
| [[Tikhonov Regularization]] | blurred   | cheap   |
| Total Variation            | preserved | moderate |

````tabs
tab: Tikhonov
```julia
reconstruct(problem, U, I, Tikhonov(1e-3); σ_init)
```
tab: TV
```julia
reconstruct(problem, U, I, TotalVariation(1e-3; ε = 1e-4); σ_init)
```
````
