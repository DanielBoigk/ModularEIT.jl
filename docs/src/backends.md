# Finite element back ends

```@meta
CurrentModule = ModularEITFerrite
```

ModularEIT itself contains no finite element code. The generic layer — electrode models,
[`ForwardModel`](@ref), objectives, regularizers, optimizers, linear solvers and synthetic data —
works with any discretization that implements a small contract. The discretization comes from a
back end package, which is loaded next to ModularEIT:

| Back end | Package | Finite element library |
|:--|:--|:--|
| Ferrite | `ModularEITFerrite` (in `lib/ModularEITFerrite`) | [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) (fork with triangle bisection) |

```julia
using ModularEIT, ModularEITFerrite, Ferrite

disc = FerriteDiscretization(generate_grid(Triangle, (16, 16)))
```

Only the back end that is used has to be installed; the others are separate packages.

```@docs
ModularEITFerrite
```

## The contract

A back end defines a subtype of [`AbstractDiscretization`](@ref) and adds methods to these
functions and constructors:

| Purpose | Functions |
|:--|:--|
| Sizes | [`ndofs_u`](@ref), [`ndofs_σ`](@ref) |
| Matrices | [`FEMatrices`](@ref)`(disc)`, [`ConductivityTensor`](@ref)`(disc; pattern, to_device)`, [`assemble_weighted_stiffness`](@ref) |
| Functions on the mesh | [`interpolate_function`](@ref), [`l2_project`](@ref), [`fe_inner`](@ref), [`fe_norm`](@ref), [`total_variation`](@ref), [`lumped_mass`](@ref) |
| Electrodes and forward model | [`angular_electrodes`](@ref), [`electrode_length`](@ref), [`transfer_electrodes`](@ref), [`ForwardModel`](@ref)`(disc, model)` for the electrode models |
| Regularizers | [`TikhonovRegularizer`](@ref)`(disc; kind)`, [`TotalVariationRegularizer`](@ref)`(disc; ε)` with `objective_value`, `value_and_gradient!`, `gauss_newton_hessian`, `prox!` |
| Fast preconditioners (optional) | [`structured_grid`](@ref), [`polar_structure`](@ref) |
| Pixel parametrizations (optional) | [`pixel_image`](@ref) |

The electrodes of [`GapModel`](@ref) and [`CompleteElectrodeModel`](@ref) are stored in the back
end's representation (for Ferrite: vectors of `FacetIndex`). Everything else the generic layer
needs is in the data types [`FEMatrices`](@ref), [`ConductivityTensor`](@ref) and
[`ForwardModel`](@ref), which are back end independent.

```@docs
total_variation!
assemble_weighted_stiffness
```
