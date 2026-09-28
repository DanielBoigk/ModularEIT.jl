# Electrode Models & Forward Problem

```@meta
CurrentModule = ModularEIT
```

An electrode model says how current enters and where voltage is measured. A
[`ForwardModel`](@ref) turns it into an injection matrix ``P``, a measurement matrix ``Q`` and
(for the complete electrode model) extra unknowns for the electrode voltages. Every model has a
current-driven (Neumann) and a voltage-driven (Dirichlet) forward problem. Injection and
measurement sites may differ. Theory: wiki articles *Electrode Models*, *Point Electrode Model*,
*Gap Model*, *Shunt Model*, *Complete Electrode Model*, *Discrete Electrode Models* and
*Measurement Protocols*.

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
