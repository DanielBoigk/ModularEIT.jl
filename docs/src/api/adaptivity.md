# Adaptive Meshing

```@meta
CurrentModule = ModularEIT
```

Adaptive refinement of quadrilateral (and hexahedral) meshes with Ferrite's AMR and of linear
triangle meshes with newest vertex bisection. Refined quadrilateral meshes have hanging nodes;
[`FerriteDiscretization`](@ref) condenses the conformity constraints, so forward models,
objectives and solvers work unchanged. Bisection keeps triangle meshes conforming (also
continuous σ spaces work) and carries facet and cell sets over. Theory: wiki articles
*Adaptive Meshing in EIT*, *Hanging Nodes*, *Newest Vertex Bisection*, *Residual Estimator for
the Conductivity Equation*, *Zienkiewicz-Zhu Estimator*, *Goal-Oriented Error Estimation* and
*Dörfler Marking*.

```julia
am = AdaptiveMesh(generate_grid(Quadrilateral, (16, 16)); maxlevel = 6)
for step in 1:10
    disc = FerriteDiscretization(current_grid(am))
    fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16), 1e-3))
    σ = interpolate_function(disc, σfun)
    I = trigonometric_patterns(fm, 8)
    _, X = forward_neumann(fm, σ, I)
    η = goal_oriented_indicator(disc, fm, σ, X; estimator = :residual, currents = I)
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
residual_indicator
flux_recovery_indicator
goal_oriented_indicator
jump_indicator
dorfler_marking
transfer_conductivity
```
