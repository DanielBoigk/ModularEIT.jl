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
Pkg.add(url = "https://github.com/DanielBoigk/ModularEIT.jl", subdir = "lib/ModularEITFerrite")
# or, for the Gridap back end (no Ferrite fork needed):
Pkg.add(url = "https://github.com/DanielBoigk/ModularEIT.jl", subdir = "lib/ModularEITGridap")
```

ModularEIT has no finite element code of its own; the discretization comes from a back end
package in `lib/` of this repository: `ModularEITFerrite`
([Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)) or `ModularEITGridap`
([Gridap.jl](https://github.com/gridap/Gridap.jl)). Load ModularEIT with one of them, e.g.
`using ModularEIT, ModularEITFerrite`.

ModularEIT depends on a fork of [Krylov.jl](https://github.com/DanielBoigk/Krylov.jl) (block
conjugate gradients with null-space projection), and the Ferrite back end on a fork of
[Ferrite.jl](https://github.com/DanielBoigk/Ferrite.jl) (newest vertex bisection with coarsening
for triangle meshes). Add the forks first: Julia uses the `[sources]` entries of a package only
when it is the active project, so `Pkg.add` alone would install the registered versions, and
ModularEIT would fail to load.

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
