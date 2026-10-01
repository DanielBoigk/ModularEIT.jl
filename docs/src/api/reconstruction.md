# Reconstruction Problems

```@meta
CurrentModule = ModularEIT
```

An [`EITProblem`](@ref) bundles everything a reconstruction needs — discretization, forward
model, data term, objective (data misfit plus regularization, optionally in parameters
`σ = P θ`), the current iterate and the history of the runs — and [`reconstruct!`](@ref) runs an
optimizer on it. Repeated calls continue from the current iterate, e.g. a cheap method first and
a more accurate one after it, or more iterations after inspecting the result. With a noise
model, every run stops by the discrepancy principle on the data misfit.

```julia
using ModularEIT, ModularEITFerrite, Ferrite

disc = FerriteDiscretization(generate_grid(Triangle, (32, 32)))
prob = EITProblem(disc, CompleteElectrodeModel(angular_electrodes(disc, 16), 0.05), currents, voltages;
                  noise = RelativeGaussianNoise(0.01),
                  regularization = (1e-4 => TotalVariationRegularizer(disc; ε = 1e-2),))
reconstruct!(prob, LBFGS(); maxiter = 20)        # coarse start
reconstruct!(prob, GaussNewton(); maxiter = 30)   # continues from there
σ = solution(prob)
```

The problem acts as an objective in its unknowns: `objective_value(prob, θ)` and
`value_and_gradient!(g, prob, θ)` evaluate the full objective, and with ChainRulesCore loaded
`objective_value(prob, θ)` has the same reverse differentiation rule as the objective (see
[Automatic differentiation](@ref)).

```@docs
EITProblem
reconstruct!
solution
data_misfit
```
