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

Theory: wiki articles *Adjoint State Method* and *Adjoint Method for the Dirichlet Problem*.

```@docs
AdjointStateObjective
residual!
residual_and_jacobian!
n_residual
```

## Kohn–Vogelius

Theory: wiki article *Kohn-Vogelius Functional*.

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

## Index

```@index
```
