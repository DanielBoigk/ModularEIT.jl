# Getting Started

This page walks through a complete (mock) reconstruction.

## Build a mesh and place electrodes

```@example tour
using ModularEIT

mesh = circle_mesh(32)
electrodes = ring_electrodes(mesh, 8)
(nnodes(mesh), nelements(mesh), length(electrodes))
```

## Simulate measurements

Inject current between two adjacent electrodes and solve the forward problem for
a homogeneous conductivity ``\sigma \equiv 2``:

```@example tour
problem = ForwardProblem(mesh, electrodes)
I = [1.0, -1.0, 0, 0, 0, 0, 0, 0]
U = solve_forward(problem, fill(2.0, nelements(mesh)), I)
```

## Reconstruct

Start from ``\sigma \equiv 1`` and minimize the Tikhonov-regularized data misfit:

```@example tour
result = reconstruct(problem, U, I, Tikhonov(1e-3); σ_init=ones(nelements(mesh)))
result.residuals[end]
```

Swap the regularizer to use total variation instead:

```@example tour
result_tv = reconstruct(problem, U, I, TotalVariation(1e-3); σ_init=ones(nelements(mesh)))
result_tv.residuals[end]
```
