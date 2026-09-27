# ModularEIT.jl

*Modular building blocks for Electrical Impedance Tomography in Julia.*

!!! warning "Mock documentation"
    This documentation and the code it describes are placeholders used to set up the
    documentation pipeline. Expect everything to change.

```@docs
ModularEIT
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/DanielBoigk/ModularEIT.jl")
```

## Package overview

| Component       | Types / functions                                         |
|:----------------|:----------------------------------------------------------|
| Geometry        | [`EITMesh`](@ref), [`circle_mesh`](@ref), [`Electrode`](@ref) |
| Forward problem | [`ForwardProblem`](@ref), [`solve_forward`](@ref), [`jacobian`](@ref) |
| Regularization  | [`Tikhonov`](@ref), [`TotalVariation`](@ref)              |
| Reconstruction  | [`reconstruct`](@ref), [`ReconstructionResult`](@ref)    |

## Theory wiki

The mathematical background (the Complete Electrode Model, regularization theory, …)
lives in an Obsidian vault that is published alongside this documentation.

```@raw html
<p><a href="wiki/">Open the ModularEIT theory wiki →</a></p>
```
