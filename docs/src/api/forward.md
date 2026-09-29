# Electrode Models & Forward Problem

```@meta
CurrentModule = ModularEIT
```

An electrode model says how current enters and where voltage is measured. A
[`ForwardModel`](@ref) turns it into an injection matrix ``P``, a measurement matrix ``Q`` and
(for the complete electrode model) extra unknowns for the electrode voltages. Every model has a
current-driven (Neumann) and a voltage-driven (Dirichlet) forward problem. Injection and
measurement sites may differ. Theory: wiki articles [Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Electrode-Models), [Point Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Point-Electrode-Model),
[Gap Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Gap-Model), [Shunt Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Shunt-Model), [Complete Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Complete-Electrode-Model), [Discrete Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Elements/Discrete-Electrode-Models) and
[Measurement Protocols](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Measurement-Protocols).

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

Truncating the pairs at the noise level regularizes in data space. It is the difference from a
reference conductivity whose singular values decay below the noise (for relative noise the
absolute data have signal/noise ≈ 1/δ in every pattern):

```julia
p = pattern_svd(disc, fm, currents, voltages; metric = :L2, noise, reference = ones(ndofs_σ(disc)))
t = truncate_patterns(p; τ = 2)                 # pairs above twice their noise level
obj = AdjointStateObjective(fm, t.currents, t.voltages)
res = minimize(obj, σ₀, GaussNewton(); ftarget = discrepancy_target(obj, t.noise))
```

```@docs
reground
pattern_svd
truncate_patterns
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Electrode-Models)
- [Complete Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Complete-Electrode-Model)
- [Gap Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Gap-Model)
- [Shunt Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Shunt-Model)
- [Point Electrode Model](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Point-Electrode-Model)
- [Current Patterns](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/01-Foundations-of-EIT/Current-Patterns)
- [Neumann Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/02-The-Forward-Problem/Neumann-Problem)
- [Dirichlet Problem](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/02-The-Forward-Problem/Dirichlet-Problem)
- [Forward Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/02-The-Forward-Problem/Forward-Map)
- [Neumann-to-Dirichlet Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/02-The-Forward-Problem/Neumann-to-Dirichlet-Map)
- [Dirichlet-to-Neumann Map](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/02-The-Forward-Problem/Dirichlet-to-Neumann-Map)
- [Truncated SVD Regularization](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Truncated-SVD-Regularization)
- [Discrete Electrode Models](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Elements/Discrete-Electrode-Models)
- [Grounding of the Potential](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Elements/Grounding-of-the-Potential)
- [State Equation](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Adjoint-Gradients/State-Equation)
