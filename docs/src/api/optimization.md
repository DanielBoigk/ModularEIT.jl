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
```

Objectives deliver coefficient gradients (dual vectors). The gradient representation (Riesz
map, e.g. the L² gradient ``M_\sigma^{-1} \nabla J``) is an option of the first-order methods,
so data term and regularizers are always added in the same representation.

Theory: wiki articles *Tikhonov Regularization*, *Smoothed Total Variation*, *Gauss-Newton
Method*, *Levenberg-Marquardt Method*, *L-BFGS*, *L-BFGS-B*, *Line Search*, *Box Constraints on
Conductivity*, *Stopping Criteria*, *Gradient Representation and the Riesz Map*.

## Regularizers

```@docs
AbstractRegularizer
TikhonovRegularizer
TotalVariationRegularizer
gauss_newton_hessian
RegularizedObjective
```

## Optimizers

```@docs
AbstractOptimizer
minimize
OptimizationState
GradientDescent
LBFGS
GaussNewton
```

## Index

```@index
Pages = ["optimization.md"]
```
