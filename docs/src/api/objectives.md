# Objectives

```@meta
CurrentModule = ModularEIT
```

Objectives evaluate a reconstruction functional and its gradient for a conductivity ``\sigma``.
All buffers are allocated when the objective is built; the linear solver is swappable
([`DirectSolver`](@ref), [`BlockCGSolver`](@ref)) and the gradient representation is chosen with
an [`AbstractRieszMap`](@ref).

```@docs
AbstractObjective
objective_value
value_and_gradient!
```

## Adjoint-state least squares

Theory: wiki articles [Adjoint State Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Adjoint-State-Method) and [Adjoint Method for the Dirichlet Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Adjoint-Method-for-the-Dirichlet-Problem).

```@docs
AdjointStateObjective
residual!
residual_and_jacobian!
n_residual
```

## Matrix-free Jacobians

For problems whose Jacobian does not fit into memory: `J * v` costs one linearized forward
solve per pattern, `J' * w` one adjoint solve per pattern (both reuse the factorization of the
forward problem). Column norms and the Gram matrix `JᵀJ` are accumulated from row blocks of
the Jacobian without storing it. `GaussNewton(; linear_solver = :cg)` uses these.

```julia
J = jacobian_operator(obj, θ)        # AdjointStateObjective or ParametrizedObjective
y = J * v; g = J' * w
s = jacobian_column_norms(obj, θ)    # sensitivities
G, g = jacobian_gram(obj, θ)         # JᵀJ and Jᵀr
```

```@docs
jacobian_operator
JacobianOperator
ParametrizedJacobian
jacobian_column_norms
jacobian_gram
```

## Kohn–Vogelius

Theory: wiki article [Kohn-Vogelius Functional](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Kohn-Vogelius-Functional).

```@docs
KohnVogeliusObjective
boundary_error
pattern_values
```

## Misfit metrics

```@docs
AbstractMisfit
SquaredEuclidean
WeightedSquaredEuclidean
ProjectedMisfit
```

## Reserved types

```@docs
AbstractEITProblem
AbstractSolutionState
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Linearized EIT and the Sensitivity Kernel](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/03-The-Inverse-Problem/Linearized-EIT-and-the-Sensitivity-Kernel)
- [Data Fidelity Terms](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Data-Fidelity-Terms)
- [Adjoint State Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Adjoint-State-Method)
- [State Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/State-Equation)
- [Adjoint Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Adjoint-Equation)
- [Adjoint Method for the Dirichlet Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Adjoint-Method-for-the-Dirichlet-Problem)
- [Functional Derivative of the Data Misfit](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Functional-Derivative-of-the-Data-Misfit)
- [Kohn-Vogelius Functional](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/Kohn-Vogelius-Functional)
- [Gauss-Newton Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/09-Optimization/Gauss-Newton-Method)

## Index

```@index
```
