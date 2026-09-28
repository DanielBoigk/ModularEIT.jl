# Adaptive Meshing

```@meta
CurrentModule = ModularEIT
```

Adaptive refinement of quadrilateral (and hexahedral) meshes with Ferrite's AMR. Refined meshes
have hanging nodes; [`FerriteDiscretization`](@ref) condenses the conformity constraints, so
forward models, objectives and solvers work unchanged. Triangle meshes are accepted with a
warning and not refined (bisection refinement is not implemented yet). Theory: wiki articles
*Adaptive Meshing in EIT*, *Hanging Nodes*, *Zienkiewicz-Zhu Estimator*, *Goal-Oriented Error
Estimation* and *Dörfler Marking*.

```julia
am = AdaptiveMesh(generate_grid(Quadrilateral, (16, 16)); maxlevel = 6)
for step in 1:10
    disc = FerriteDiscretization(current_grid(am))
    fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16), 1e-3))
    σ = interpolate_function(disc, σfun)
    _, X = forward_neumann(fm, σ, trigonometric_patterns(fm, 8))
    η = goal_oriented_indicator(disc, fm, σ, X)
    η[cell_levels(am) .>= max_level(am)] .= 0
    refine_mesh!(am, dorfler_marking(η, 0.3))
end
```

```@docs
AdaptiveMesh
current_grid
refine_mesh!
coarsen_mesh!
cell_levels
max_level
is_nonconforming
flux_recovery_indicator
goal_oriented_indicator
jump_indicator
dorfler_marking
transfer_conductivity
```
