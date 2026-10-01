# Getting Started

A forward simulation and the two objectives on a small quadrilateral mesh (the kind of mesh a
pixel image gives).

## Discretization and electrodes

```@example tour
using ModularEIT, ModularEITFerrite, Ferrite

grid = generate_grid(Quadrilateral, (24, 24))          # [-1, 1]², Q1 potential, Q0 conductivity
disc = FerriteDiscretization(grid)
electrodes = angular_electrodes(disc, 16; coverage = 0.5)
fm = ForwardModel(disc, CompleteElectrodeModel(electrodes, 0.05))
(ndofs_u(disc), ndofs_σ(disc), n_inject(fm), n_measure(fm))
```

## Simulate data

A conductive inclusion, trigonometric current patterns, measured electrode voltages:

```@example tour
σ_true = interpolate_function(disc, x -> hypot(x[1] - 0.3, x[2]) < 0.35 ? 3.0 : 1.0)
currents = trigonometric_patterns(fm, 4)                # 8 patterns
voltages, _ = forward_neumann(fm, σ_true, currents)
size(voltages)
```

## Objectives and gradients

Least squares through the adjoint state method, with the L² gradient (Riesz map with the
σ mass matrix):

```@example tour
σ0 = ones(ndofs_σ(disc))
obj = AdjointStateObjective(fm, currents, voltages; gradient = L2Gradient(FEMatrices(disc)))
g = zeros(ndofs_σ(disc))
J = value_and_gradient!(g, obj, σ0)
(J, objective_value(obj, σ_true))
```

The Kohn–Vogelius functional needs no adjoint solve:

```@example tour
kv = KohnVogeliusObjective(fm, currents, voltages)
(value_and_gradient!(g, kv, σ0), objective_value(kv, σ_true))
```

Swap the linear solver without changing anything else:

```@example tour
obj_cg = AdjointStateObjective(fm, currents, voltages; solver = BlockCGSolver(; preconditioner = :amg))
objective_value(obj_cg, σ0) ≈ objective_value(obj, σ0)
```
