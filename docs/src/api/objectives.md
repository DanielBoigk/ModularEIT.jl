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

Theory: wiki articles [Adjoint State Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Adjoint-State-Method) and [Adjoint Method for the Dirichlet Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Adjoint-Method-for-the-Dirichlet-Problem).

```@docs
AdjointStateObjective
residual!
residual_and_jacobian!
n_residual
```

## Kohn–Vogelius

Theory: wiki article [Kohn-Vogelius Functional](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Kohn-Vogelius-Functional).

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
```

## Reserved types

```@docs
AbstractEITProblem
AbstractSolutionState
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Linearized EIT and the Sensitivity Kernel](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/02-Inverse-Problem/Linearized-EIT-and-the-Sensitivity-Kernel)
- [Data Fidelity Terms](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/03-Regularization/Data-Fidelity-Terms)
- [Adjoint State Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Adjoint-State-Method)
- [State Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/State-Equation)
- [Adjoint Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Adjoint-Equation)
- [Adjoint Method for the Dirichlet Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Adjoint-Method-for-the-Dirichlet-Problem)
- [Functional Derivative of the Data Misfit](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Functional-Derivative-of-the-Data-Misfit)
- [Kohn-Vogelius Functional](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/Kohn-Vogelius-Functional)
- [Gauss-Newton Method](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Optimization/Gauss-Newton-Method)

## Index

```@index
```
