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
Pkg.add(url = "https://github.com/DanielBoigk/Ferrite.jl", rev = "adaptive-triangular")
Pkg.add(url = "https://github.com/DanielBoigk/Krylov.jl", rev = "block-cg")
Pkg.add(url = "https://github.com/DanielBoigk/ModularEIT.jl")
```

ModularEIT depends on forks of [Ferrite.jl](https://github.com/DanielBoigk/Ferrite.jl) (newest
vertex bisection with coarsening for triangle meshes) and
[Krylov.jl](https://github.com/DanielBoigk/Krylov.jl) (block conjugate gradients with null-space
projection). Add the forks first: Julia uses the `[sources]` entries of a package only when it
is the active project, so `Pkg.add` of ModularEIT alone would install the registered versions,
and ModularEIT would fail to load.

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
