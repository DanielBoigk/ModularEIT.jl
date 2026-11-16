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
Pkg.Registry.add(RegistrySpec(url = "https://github.com/MathStruct/Registry.git"))
Pkg.add(["ModularEIT", "ModularEITFerrite"])
# or, for the Gridap back end:
Pkg.add(["ModularEIT", "ModularEITGridap"])
```

ModularEIT has no finite element code of its own; the discretization comes from a back end
package in `lib/` of this repository: `ModularEITFerrite`
([Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)) or `ModularEITGridap`
([Gridap.jl](https://github.com/gridap/Gridap.jl)). Load ModularEIT with one of them, e.g.
`using ModularEIT, ModularEITFerrite`.

ModularEIT depends on [KrylovBlockCG.jl](https://github.com/DanielBoigk/KrylovBlockCG.jl) (block
conjugate gradients with null-space projection, for Krylov.jl), and the Ferrite back end on
[FerriteAdaptiveTriangular.jl](https://github.com/DanielBoigk/FerriteAdaptiveTriangular.jl)
(newest vertex bisection with coarsening for triangle meshes, for Ferrite.jl). Both work with
the released Krylov.jl and Ferrite.jl. All of them are in the
[MathStruct registry](https://github.com/MathStruct/Registry), which the first line adds.

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
