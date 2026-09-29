# Electrode Models & Forward Problem

```@meta
CurrentModule = ModularEIT
```

An electrode model says how current enters and where voltage is measured. A
[`ForwardModel`](@ref) turns it into an injection matrix ``P``, a measurement matrix ``Q`` and
(for the complete electrode model) extra unknowns for the electrode voltages. Every model has a
current-driven (Neumann) and a voltage-driven (Dirichlet) forward problem. Injection and
measurement sites may differ. Theory: wiki articles [Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Electrode-Models), [Point Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Point-Electrode-Model),
[Gap Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Gap-Model), [Shunt Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Shunt-Model), [Complete Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Complete-Electrode-Model), [Discrete Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Discrete-Electrode-Models) and
[Measurement Protocols](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Measurement-Protocols).

```@docs
AbstractElectrodeModel
ContinuumModel
PointElectrodeModel
GapModel
CompleteElectrodeModel
angular_electrodes
electrode_length
```

## Forward model

```@docs
AbstractForwardModel
ForwardModel
system_matrix!
n_inject
trigonometric_patterns
forward_neumann
forward_dirichlet
```

## Regrounding and pattern SVD

Voltages under different groundings differ by one constant per pattern, so they can be shifted
back and forth exactly. Linear combinations of measured pairs are measured pairs as well; the
SVD rotates them into orthonormal patterns, either in the Euclidean inner product (matches the
nodal-sum ground) or in L²(Γ) (matches the boundary-mean ground, independent of the mesh).

```@docs
reground
pattern_svd
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Electrode-Models)
- [Complete Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Complete-Electrode-Model)
- [Gap Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Gap-Model)
- [Shunt Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Shunt-Model)
- [Point Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Point-Electrode-Model)
- [Current Patterns](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Current-Patterns)
- [Neumann Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Neumann-Problem)
- [Dirichlet Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Dirichlet-Problem)
- [Forward Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Forward-Map)
- [Neumann-to-Dirichlet Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Neumann-to-Dirichlet-Map)
- [Dirichlet-to-Neumann Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Physics-and-Forward-Problem/Dirichlet-to-Neumann-Map)
- [Discrete Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Discrete-Electrode-Models)
- [Grounding of the Potential](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Grounding-of-the-Potential)
- [State Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/05-Adjoint-Gradients/State-Equation)
