"""
    Regularizer

Abstract supertype of all regularization functionals ``R(\\sigma)``.

A subtype `R <: Regularizer` must implement [`penalty`](@ref) and
[`gradient`](@ref).
"""
abstract type Regularizer end

"""
    penalty(R::Regularizer, σ::AbstractVector) -> Real

Value of the regularization functional at `σ`.
"""
function penalty end

"""
    gradient(R::Regularizer, σ::AbstractVector) -> Vector

Gradient of the regularization functional at `σ`.
"""
function gradient end

"""
    Tikhonov(α; σ₀=0.0)

Tikhonov regularization ``R(\\sigma) = \\frac{\\alpha}{2} \\lVert \\sigma - \\sigma_0 \\rVert_2^2``.

# Examples
```jldoctest
julia> penalty(Tikhonov(2.0), [1.0, 2.0])
5.0
```
"""
struct Tikhonov{T<:Real} <: Regularizer
    α::T
    σ₀::T
end
Tikhonov(α::Real; σ₀::Real=0.0) = Tikhonov(promote(α, σ₀)...)

penalty(R::Tikhonov, σ::AbstractVector) = R.α / 2 * sum(abs2, σ .- R.σ₀)
gradient(R::Tikhonov, σ::AbstractVector) = R.α .* (σ .- R.σ₀)

"""
    TotalVariation(α; ε=1e-6)

Smoothed one-dimensional total variation

```math
R(\\sigma) = \\alpha \\sum_k \\sqrt{(\\sigma_{k+1} - \\sigma_k)^2 + \\varepsilon^2}.
```

The mock ignores the mesh and uses the element ordering as neighbourhood.
"""
struct TotalVariation{T<:Real} <: Regularizer
    α::T
    ε::T
end
TotalVariation(α::Real; ε::Real=1e-6) = TotalVariation(promote(α, ε)...)

penalty(R::TotalVariation, σ::AbstractVector) = R.α * sum(sqrt.(diff(σ) .^ 2 .+ R.ε^2))

function gradient(R::TotalVariation, σ::AbstractVector)
    d = diff(σ)
    w = d ./ sqrt.(d .^ 2 .+ R.ε^2)
    g = zeros(float(eltype(σ)), length(σ))
    g[1:end-1] .-= w
    g[2:end] .+= w
    return R.α .* g
end
