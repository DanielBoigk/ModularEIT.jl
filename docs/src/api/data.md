# Synthetic Data & Noise

```@meta
CurrentModule = ModularEIT
```

Phantoms are functions ``x \mapsto \sigma(x)`` that do not depend on a mesh. They are put on a
discretization by [`conductivity`](@ref) (cell averages for piecewise constants). This makes it
easy to simulate data on a finer mesh than the reconstruction mesh, with the same physical
electrodes ([`transfer_electrodes`](@ref)), and so avoid the inverse crime:

```julia
coarse, fine = FerriteDiscretization(grid_coarse), FerriteDiscretization(grid_fine)
els  = angular_electrodes(coarse, 16; coverage = 0.5)
fm   = ForwardModel(coarse, CompleteElectrodeModel(els, 0.1))
fm_f = ForwardModel(fine, CompleteElectrodeModel(transfer_electrodes(coarse, els, fine), 0.1))

phantom = random_inclusions(rng; count = 1:3)            # or gaussian_random_field, image_phantom
currents = trigonometric_patterns(fm, 7)
noise = RelativeGaussianNoise(0.01)
sim = simulate_data(fine, fm_f, phantom, currents; noise, rng)

data = AdjointStateObjective(fm, currents, sim.data)
res = minimize(RegularizedObjective(data, 1e-4 => TotalVariationRegularizer(coarse; ε = 1e-2)),
               ones(ndofs_σ(coarse)), GaussNewton(); lower = 0.05,
               ftarget = discrepancy_target(data, noise))   # discrepancy principle
```

The discrepancy target only accounts for the instrument noise. When the modelling error (for
example of a coarse reconstruction mesh) is larger than the noise, the data have to be explained
beyond what the model can represent, and the reconstruction deteriorates. Compare the model error
`norm(sim.clean - forward_neumann(fm, conductivity(coarse, phantom), currents)[1])` with the noise
level before choosing the target.

## Noise models

```@docs
AbstractNoiseModel
GaussianNoise
RelativeGaussianNoise
SourceMeterNoise
add_noise
expected_squared_error
discrepancy_target
perturb_boundary_operator
```

## Modelling errors

```@docs
perturb_contact_impedance
electrode_angles
transfer_electrodes
```

## Phantoms

```@docs
AbstractInclusion
Circle
Ellipse
Polygon
InclusionPhantom
random_inclusions
PixelFunction
image_phantom
gaussian_random_field
TransformedPhantom
lognormal_phantom
levelset_phantom
```

## Simulation

```@docs
conductivity
simulate_data
```

## Image corruption

```@docs
corrupt_image
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Choosing the Regularization Parameter](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/03-Regularization/Choosing-the-Regularization-Parameter)
- [Spectral Sobolev Norms on Rectangles](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/04-Finite-Element-Discretization/Spectral-Sobolev-Norms-on-Rectangles)
- [Stopping Criteria](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/06-Optimization/Stopping-Criteria)
- [Noise Models for EIT Data](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Data-and-Noise/Noise-Models-for-EIT-Data)
- [Synthetic Conductivity Data](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Data-and-Noise/Synthetic-Conductivity-Data)
- [Inverse Crime](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Data-and-Noise/Inverse-Crime)
- [Spectral Image Corruption](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/07-Data-and-Noise/Spectral-Image-Corruption)

## Index

```@index
Pages = ["data.md"]
```
