# Regularization & Optimization

```@meta
CurrentModule = ModularEIT
```

A reconstruction minimizes a data objective plus regularizers,
``J(\sigma) = J_\text{data}(\sigma) + \sum_k \alpha_k R_k(\sigma)``, optionally subject to
bounds ``\sigma_\text{lo} \le \sigma \le \sigma_\text{hi}``:

```julia
disc = FerriteDiscretization(grid)
mats = FEMatrices(disc)
data = AdjointStateObjective(fm, currents, voltages)
obj  = RegularizedObjective(data, 1e-4 => TotalVariationRegularizer(disc; ε = 1e-2))

res = minimize(obj, ones(ndofs_σ(disc)), GaussNewton(); lower = 1e-2, maxiter = 30)
res = minimize(obj, ones(ndofs_σ(disc)), LBFGS(riesz = L2Gradient(mats)); lower = 1e-2)
res.σ, res.status, res.history

# exact (non-smooth) TV through its proximal operator
tv = TotalVariationRegularizer(disc; ε = 0)
res = minimize(data, σ₀, ADMM(1e-4 => tv; weights = lumped_mass(disc), inner = GaussNewton()); lower = 1e-2)
res = minimize(data, σ₀, ProximalGradient(1e-4 => tv; weights = lumped_mass(disc)); lower = 1e-2)
```

Objectives deliver coefficient gradients (dual vectors). The gradient representation (Riesz
map, e.g. the L² gradient ``M_\sigma^{-1} \nabla J``) is an option of the first-order methods,
so data term and regularizers are always added in the same representation.

Theory: wiki articles [Tikhonov Regularization](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Tikhonov-Regularization), [Smoothed Total Variation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Smoothed-Total-Variation), [Gauss-Newton
Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Gauss-Newton-Method), [Levenberg-Marquardt Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Levenberg-Marquardt-Method), [L-BFGS](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/L-BFGS), [L-BFGS-B](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/L-BFGS-B), [Line Search](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Line-Search), [Proximal Operator](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Proximal-Operator), [ADMM](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/ADMM), [Nested ADMM Reconstruction](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Nested-ADMM-Reconstruction),
[Chambolle-Pock Algorithm](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Chambolle-Pock-Algorithm), [Box Constraints on Conductivity](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Box-Constraints-on-Conductivity), [Stopping Criteria](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Stopping-Criteria), [Gradient Representation and the Riesz Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Gradient-Representation-and-the-Riesz-Map).

## Regularizers

```@docs
AbstractRegularizer
TikhonovRegularizer
TotalVariationRegularizer
gauss_newton_hessian
RegularizedObjective
```

## Proximal operators

```@docs
prox!
ProximalMap
lumped_mass
```

## Optimizers

```@docs
AbstractOptimizer
minimize
OptimizationState
GradientDescent
LBFGS
GaussNewton
ProximalGradient
ADMM
```

## Truncated SVD

Singular value decomposition of the Jacobian (modes ordered by how well the data determine
them, i.e. by depth) and Gauss–Newton with truncated-SVD steps: regularisation by projection,
no penalty, stopped by the discrepancy principle.

```julia
obj = ParametrizedObjective(AdjointStateObjective(fm, currents, voltages), pixels)
js  = jacobian_svd(obj, θ₀)                       # (U, s, V, r), J = U S Vᵀ
res = minimize(obj, θ₀, TruncatedGaussNewton(; rtol = 1e-2); lower = 0.05,
               ftarget = discrepancy_target(obj.obj, noise))
sp  = SubspaceParametrization(pixels, jacobian_basis(obj, θ₀, 40))   # data-optimal subspace
```

```@docs
jacobian_svd
jacobian_basis
TruncatedGaussNewton
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Variational Regularization](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Variational-Regularization)
- [Tikhonov Regularization](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Tikhonov-Regularization)
- [Total Variation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Total-Variation)
- [Smoothed Total Variation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Smoothed-Total-Variation)
- [Iterative Reconstruction Loop](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Iterative-Reconstruction-Loop)
- [Gauss-Newton Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Gauss-Newton-Method)
- [Levenberg-Marquardt Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Levenberg-Marquardt-Method)
- [Line Search](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Line-Search)
- [L-BFGS](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/L-BFGS)
- [L-BFGS-B](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/L-BFGS-B)
- [Proximal Operator](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Proximal-Operator)
- [ADMM](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/ADMM)
- [Chambolle-Pock Algorithm](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Chambolle-Pock-Algorithm)
- [Nested ADMM Reconstruction](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Nested-ADMM-Reconstruction)
- [Box Constraints on Conductivity](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Box-Constraints-on-Conductivity)
- [Stopping Criteria](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Stopping-Criteria)
- [Plug-and-Play Priors](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/11-Learned-Priors/Plug-and-Play-Priors)

## Index

```@index
Pages = ["optimization.md"]
```
