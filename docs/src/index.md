# ModularEIT.jl

*Modular building blocks for Electrical Impedance Tomography in Julia.*

!!! warning "Work in progress"
    The package is being rebuilt; interfaces may still change.

```@docs
ModularEIT
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/DanielBoigk/ModularEIT.jl")
```

## Package overview

| Component           | Types / functions                                                          |
|:--------------------|:---------------------------------------------------------------------------|
| Discretization      | [`FerriteDiscretization`](@ref), [`FEMatrices`](@ref), [`ConductivityTensor`](@ref) |
| Electrode models    | [`ContinuumModel`](@ref), [`PointElectrodeModel`](@ref), [`GapModel`](@ref), [`CompleteElectrodeModel`](@ref) |
| Forward problem     | [`ForwardModel`](@ref), [`forward_neumann`](@ref), [`forward_dirichlet`](@ref) |
| Objectives          | [`AdjointStateObjective`](@ref), [`KohnVogeliusObjective`](@ref)            |
| Linear solvers      | [`DirectSolver`](@ref), [`BlockCGSolver`](@ref), [`pbcg`](@ref), [`projected_cholesky`](@ref) |

## Theory wiki

The mathematical background (the Complete Electrode Model, regularization theory, …)
lives in an Obsidian vault that is published alongside this documentation.

```@raw html
<p><a href="wiki/">Open the ModularEIT theory wiki →</a></p>
```
