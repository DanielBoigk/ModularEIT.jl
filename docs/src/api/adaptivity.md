# Adaptive Meshing

```@meta
CurrentModule = ModularEIT
```

Adaptive refinement of quadrilateral (and hexahedral) meshes with Ferrite's AMR and of linear
triangle meshes with newest vertex bisection (`BisectionMesh` of the AMR module in the
[Ferrite.jl fork](https://github.com/DanielBoigk/Ferrite.jl) that ModularEIT depends on). Refined quadrilateral meshes have hanging nodes;
[`FerriteDiscretization`](@ref) condenses the conformity constraints, so forward models,
objectives and solvers work unchanged. Bisection keeps triangle meshes conforming (also
continuous σ spaces work) and carries facet and cell sets over. Theory: wiki articles
[Adaptive Meshing in EIT](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Adaptive-Meshing-in-EIT), [Hanging Nodes](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Hanging-Nodes), [Newest Vertex Bisection](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Newest-Vertex-Bisection), [Residual Estimator for
the Conductivity Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Residual-Estimator-for-the-Conductivity-Equation), [Zienkiewicz-Zhu Estimator](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Zienkiewicz-Zhu-Estimator), [Goal-Oriented Error Estimation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Goal-Oriented-Error-Estimation) and
[Dörfler Marking](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Dörfler-Marking).

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

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [L2 Projection](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Elements/L2-Projection)
- [A Posteriori Error Estimation and Adaptive Meshing](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/A-Posteriori-Error-Estimation-and-Adaptive-Meshing)
- [Adaptive Meshing in EIT](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Adaptive-Meshing-in-EIT)
- [Residual Estimator for the Conductivity Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Residual-Estimator-for-the-Conductivity-Equation)
- [Zienkiewicz-Zhu Estimator](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Zienkiewicz-Zhu-Estimator)
- [Goal-Oriented Error Estimation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Goal-Oriented-Error-Estimation)
- [Dörfler Marking](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Dörfler-Marking)
- [Hanging Nodes](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Hanging-Nodes)
- [Newest Vertex Bisection](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Newest-Vertex-Bisection)
- [Coarsening of Bisection Meshes](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Meshes-and-Geometry/Coarsening-of-Bisection-Meshes)
