# Parametrizations

```@meta
CurrentModule = ModularEITFerrite
```

The unknowns of a reconstruction need not be the finite element coefficients of the
conductivity. A parametrization ``\sigma = P\theta`` maps parameters to coefficients, and
[`ParametrizedObjective`](@ref) turns any objective into a function of ``\theta`` (gradient
``P^\top\nabla_\sigma J``, Jacobian ``(\partial r/\partial\sigma) P``), so every optimizer applies.

- [`PixelParametrization`](@ref): the parameters are the pixels of an image on a rectangle that
  contains the domain, independent of the mesh. The mesh can be refined anywhere (e.g. towards the
  boundary) while the unknowns stay pixels, e.g. for image priors.
- [`SubspaceParametrization`](@ref): pixels restricted to a subspace, e.g. low-frequency DCT modes
  ([`dct_basis`](@ref)) for a stable first stage of a coarse-to-fine reconstruction, or DCT modes
  plus free pixels along the boundary ([`boundary_band_basis`](@ref)).

```julia
pp  = PixelParametrization(disc, 64, 64)                     # disc: any 2D mesh, e.g. refined at the boundary
obj = RegularizedObjective(ParametrizedObjective(data, pp),
                           1e-3 => TotalVariationRegularizer(pp.pixel_disc; ε = 1e-2))
res = minimize(obj, ones(parameter_count(pp)), GaussNewton(); lower = 0.05)
img = pixel_image(pp, res.σ)                                  # 64 × 64, NaN outside the domain

C   = dct_basis(pp, 8)                                        # coarse stage: 64 DCT modes
sp  = SubspaceParametrization(pp, C)
res0 = minimize(ParametrizedObjective(data, sp), [64.0; zeros(63)], GaussNewton())
θ0  = C * res0.σ                                              # pixel start for the fine stage
```

Bounds on pixel parameters bound the conductivity (the weights of ``P`` are non-negative and sum to
one). Subspace parameters cannot be bounded this way: trial steps with a non-positive
conductivity are rejected ([`InfeasibleConductivityError`](@ref)).

Theory: wiki article [Parametrizations of the Conductivity](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Parametrizations-of-the-Conductivity).

```@docs
AbstractParametrization
ModularEITFerrite.PixelParametrization
SubspaceParametrization
ParametrizedObjective
conductivity(::AbstractParametrization, ::AbstractVector)
parameter_count
pixel_image
ModularEITFerrite.pixel_parameters
ModularEITFerrite.dct_basis
ModularEITFerrite.boundary_band_basis
InfeasibleConductivityError
```

## Wiki articles

Theory behind this page in the [theory wiki](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/):

- [Parametrizations of the Conductivity](https://danielboigk.github.io/ModularEIT.jl/dev/wiki/08-Regularization/Parametrizations-of-the-Conductivity)
