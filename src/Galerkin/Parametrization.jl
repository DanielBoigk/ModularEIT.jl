# Parametrizations σ = P θ of the conductivity coefficients, and objectives in the parameters.
#
# Reconstructions can then work with unknowns other than the finite element coefficients: the
# pixels of an image (PixelParametrization, independent of the mesh), or a subspace of them
# (SubspaceParametrization, e.g. low-frequency DCT modes, or DCT modes plus a band of free pixels
# along the boundary). The chain rule is linear:
#
#     ∇_θ J = Pᵀ ∇_σ J,     ∂r/∂θ = (∂r/∂σ) P,
#
# so every optimizer, and Gauss–Newton through the Jacobian, applies unchanged.

"""
    conductivity(par::AbstractParametrization, θ)

Conductivity coefficients `σ = P θ` of the parameters `θ`.
"""
conductivity(par::AbstractParametrization, θ::AbstractVector) = par.P * θ

"""
    parameter_count(par)

Number of parameters of a parametrization.
"""
parameter_count(par::AbstractParametrization) = size(par.P, 2)

"""
    SubspaceParametrization(pixels::PixelParametrization, B)

Pixel values restricted to the span of the columns of `B` (`n m × k`): `pixels = B θ`,
`σ = P_pixels B θ`. Bases: [`dct_basis`](@ref) (smooth, low-dimensional),
[`boundary_band_basis`](@ref) (free pixels along the boundary), or combinations `[B₁ B₂]`.
"""
struct SubspaceParametrization{PP, MB, MP} <: AbstractParametrization
    pixels::PP
    B::MB
    P::MP
end
function SubspaceParametrization(pixels::AbstractParametrization, B::AbstractMatrix)
    size(B, 1) == parameter_count(pixels) || throw(DimensionMismatch("B needs $(parameter_count(pixels)) rows"))
    P = pixels.P * B
    return SubspaceParametrization(pixels, B, P isa SparseMatrixCSC ? P : Matrix(P))
end

pixel_image(sp::SubspaceParametrization, θ::AbstractVector) = pixel_image(sp.pixels, sp.B * θ)

"""
    ParametrizedObjective(obj, par)

The objective `obj` as a function of the parameters `θ` of `par` (`σ = P θ`): values, gradients
`Pᵀ ∇_σ J` and, for least-squares objectives, residuals and Jacobians `(∂r/∂σ) P`, so that all
optimizers apply (bounds on pixel parameters bound σ, see [`PixelParametrization`](@ref)).
Regularizers on the parameters act on the pixel discretization, e.g.
`RegularizedObjective(ParametrizedObjective(data, pp), α => TotalVariationRegularizer(pp.pixel_disc))`.
"""
mutable struct ParametrizedObjective{O <: AbstractObjective, Par <: AbstractParametrization} <: AbstractObjective
    obj::O
    par::Par
    gσ::Vector{Float64}
    Jσ::Matrix{Float64}          # Jacobian buffer (allocated on first use)
end
function ParametrizedObjective(obj::AbstractObjective, par::AbstractParametrization)
    _require_coefficient_gradient(obj)
    return ParametrizedObjective(obj, par, zeros(size(par.P, 1)), zeros(0, 0))
end

objective_value(o::ParametrizedObjective, θ::AbstractVector) = objective_value(o.obj, conductivity(o.par, θ))

function value_and_gradient!(g::AbstractVector, o::ParametrizedObjective, θ::AbstractVector)
    J = value_and_gradient!(o.gσ, o.obj, conductivity(o.par, θ))
    mul!(g, o.par.P', o.gσ)
    return J
end

n_residual(o::ParametrizedObjective) = n_residual(o.obj)
residual!(r::AbstractVector, o::ParametrizedObjective, θ::AbstractVector) = residual!(r, o.obj, conductivity(o.par, θ))

function residual_and_jacobian!(r::AbstractVector, J::AbstractMatrix, o::ParametrizedObjective, θ::AbstractVector)
    size(o.Jσ) == (length(r), size(o.par.P, 1)) || (o.Jσ = zeros(length(r), size(o.par.P, 1)))
    residual_and_jacobian!(r, o.Jσ, o.obj, conductivity(o.par, θ))
    mul!(J, o.Jσ, o.par.P)
    return r, J
end
