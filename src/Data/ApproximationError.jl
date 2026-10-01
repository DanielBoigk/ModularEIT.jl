# Approximation-error model (Kaipio & Somersalo): the data are the predictions of an accurate
# model plus noise, d = F(σ) + e. A reconstruction model F̃ (coarser mesh, pixels, ...) misses
# them by the modelling error ε(σ) = F(σ) - F̃(σ), so d = F̃(σ) + ε + e. Modelling ε as Gaussian
# N(μ, Γ), independent of σ, with μ and Γ estimated from samples σ⁽ⁱ⁾ of the prior, the misfit
# becomes ½ ‖C^{-1/2} (r - μ)‖² with C = η² I + Γ, where r is the residual of the reconstruction
# model and μ the mean residual of the samples.
#
# With K samples the sample covariance Γ̂ has rank < K, usually far below the number m of
# residuals: a new modelling error has components outside the span of the samples, which Γ̂
# ignores and which would be divided by the (small) noise level alone. A variance floor ν² for
# these directions, estimated by leave-one-out (the part of each sample outside the span of the
# others), gives C = (η² + ν²) I + Γ̂ =: σ_d² (I + Ẽ Ẽᵀ) with Ẽ = (R - μ1ᵀ)/(σ_d √(K-1)). With
# the thin SVD Ẽ = Q S Vᵀ,
#     C^{-1/2} = σ_d⁻¹ (I - Q diag(wₖ) Qᵀ),   wₖ = 1 - 1/√(1 + sₖ²),
# so the whitening costs two thin products and no m × m matrix.

"""
    ApproximationError(R; noise, floor = :leave_one_out, rtol = 1e-10)

Gaussian model `N(μ, Γ)` of the modelling error from samples: the columns of `R` are residuals
of the reconstruction model at the true parameters of sample conductivities, for noise-free
data of the accurate model (e.g. a finer mesh). `noise` is the standard deviation `η` of the
measurement noise per residual entry. Stores the mean `μ` and the low-rank factor of the
whitening `C^{-1/2}`, `C = (η² + ν²) I + Γ̂` (see [`whiten`](@ref)), with the sample covariance
`Γ̂` and a variance floor `ν²` for the directions the samples do not span: estimated by
leave-one-out (`floor = :leave_one_out`: the mean squared part of a sample outside the span of
the other samples, per remaining dimension; conservative, since the span of the other samples is
itself perturbed), or given as a number (`floor = 0`: none).
Singular values below `rtol` times the largest are dropped.
"""
struct ApproximationError
    μ::Vector{Float64}
    Q::Matrix{Float64}
    w::Vector{Float64}
    η::Float64          # σ_d = √(noise² + ν²), the level of the isotropic part
    noise::Float64
    ν::Float64
end
function ApproximationError(R::AbstractMatrix; noise::Real, floor = :leave_one_out, rtol::Real = 1e-10)
    m, K = size(R)
    K >= 3 || throw(ArgumentError("need at least three samples"))
    noise > 0 || throw(ArgumentError("the noise level must be positive"))
    μ = vec(sum(R; dims = 2)) ./ K
    E = R .- μ
    ν = floor === :leave_one_out ? _leave_one_out_floor(R) :
        floor isa Real && floor >= 0 ? Float64(floor) : throw(ArgumentError("floor must be :leave_one_out or ≥ 0"))
    σd = sqrt(noise^2 + ν^2)
    F = svd(E ./ (σd * sqrt(K - 1)))
    k = count(>(rtol * max(first(F.S), floatmin())), F.S)
    s = F.S[1:k]
    return ApproximationError(μ, F.U[:, 1:k], 1 .- 1 ./ sqrt.(1 .+ s .^ 2), σd, Float64(noise), ν)
end

# standard deviation per dimension of the part of a sample outside the span of the others
# (leave-one-out: sample i centred with the mean of the others, projected onto their span)
function _leave_one_out_floor(R::AbstractMatrix)
    m, K = size(R)
    K - 2 < m || return 0.0
    acc = 0.0
    for i in 1:K
        others = R[:, [j for j in 1:K if j != i]]
        μo = vec(sum(others; dims = 2)) ./ (K - 1)
        Qo = qr(others .- μo).Q * Matrix(1.0I, m, K - 2)     # thin factor of the centred others
        e = R[:, i] .- μo
        acc += sum(abs2, e .- Qo * (Qo' * e))
    end
    return sqrt(acc / K / (m - (K - 2)))
end

"""
    whiten(ae, r)

The whitened residual `C^{-1/2} (r - μ)` of the [`ApproximationError`](@ref) `ae`.
"""
whiten(ae::ApproximationError, r::AbstractVector) = _whiten_linear(ae, r .- ae.μ)

# C^{-1/2} x (the linear part, symmetric)
_whiten_linear(ae::ApproximationError, x::AbstractVecOrMat) = (y = x ./ ae.η; y .- ae.Q * (ae.w .* (ae.Q' * y)))

"""
    ApproximationErrorObjective(obj, ae)

The least-squares objective `obj` with the modelling error of `ae` accounted for: residual
`C^{-1/2}(r - μ)`, so that the misfit of the true conductivity is of the size of the noise
again. Values, gradients, Jacobians (explicit and matrix-free, [`jacobian_operator`](@ref)),
column norms and Gram matrices, so that all Gauss–Newton variants apply. Stop by the
discrepancy principle with `discrepancy_target(obj; τ)` (the whitened residual has unit
variance per entry).
"""
struct ApproximationErrorObjective{O <: AbstractObjective} <: AbstractObjective
    obj::O
    ae::ApproximationError
    r::Vector{Float64}
end
function ApproximationErrorObjective(obj::AbstractObjective, ae::ApproximationError)
    n_residual(obj) == length(ae.μ) ||
        throw(DimensionMismatch("the objective has $(n_residual(obj)) residuals, the model $(length(ae.μ))"))
    return ApproximationErrorObjective(obj, ae, zeros(n_residual(obj)))
end

n_residual(o::ApproximationErrorObjective) = n_residual(o.obj)
residual!(r::AbstractVector, o::ApproximationErrorObjective, θ::AbstractVector) =
    copyto!(r, whiten(o.ae, residual!(o.r, o.obj, θ)))
objective_value(o::ApproximationErrorObjective, θ::AbstractVector) = sum(abs2, residual!(similar(o.r), o, θ)) / 2

function value_and_gradient!(g::AbstractVector, o::ApproximationErrorObjective, θ::AbstractVector)
    r = residual!(similar(o.r), o, θ)
    mul!(g, jacobian_operator(o.obj, θ)', _whiten_linear(o.ae, r))       # Jᵀ C^{-1/2} r̃
    return sum(abs2, r) / 2
end

function residual_and_jacobian!(r::AbstractVector, Jm::AbstractMatrix, o::ApproximationErrorObjective, θ::AbstractVector)
    residual_and_jacobian!(o.r, Jm, o.obj, θ)
    copyto!(r, whiten(o.ae, o.r))
    Jm .= _whiten_linear(o.ae, Jm)
    return r, Jm
end

"""
    discrepancy_target(obj::ApproximationErrorObjective; τ = 1.1)

`τ² m / 2` for `m` residuals: the expected misfit of the whitened residual.
"""
discrepancy_target(o::ApproximationErrorObjective; τ::Real = 1.1) = τ^2 * n_residual(o) / 2

# matrix-free: C^{-1/2} J
struct WhitenedJacobian{J}
    J::J
    ae::ApproximationError
end
jacobian_operator(o::ApproximationErrorObjective, θ::AbstractVector) = WhitenedJacobian(jacobian_operator(o.obj, θ), o.ae)
Base.size(J::WhitenedJacobian) = size(J.J)
Base.size(J::WhitenedJacobian, d::Integer) = size(J.J, d)
Base.eltype(::WhitenedJacobian) = Float64
Base.adjoint(J::WhitenedJacobian) = AdjointJacobianOperator(J)
Base.:*(J::WhitenedJacobian, v::AbstractVector) = mul!(zeros(size(J, 1)), J, v)
LinearAlgebra.mul!(y::AbstractVector, J::WhitenedJacobian, v::AbstractVector) = copyto!(y, _whiten_linear(J.ae, J.J * v))
LinearAlgebra.mul!(g::AbstractVector, A::AdjointJacobianOperator{<:WhitenedJacobian}, w::AbstractVector) =
    mul!(g, A.J.J', _whiten_linear(A.J.ae, w))

# Column norms and Gram matrix from row blocks of the unwhitened Jacobian: with A = J/σ_d and
# Cq = Qᵀ A (k × n), C^{-1/2} J = A - Q diag(w) Cq, so
#     (C^{-1/2} J)ᵀ (C^{-1/2} J) = AᵀA - Cqᵀ diag(2w - w²) Cq.
function _whitened_blocks(o::ApproximationErrorObjective, θ, want_gram::Bool)
    ae = o.ae
    n, k = length(θ), length(ae.w)
    a2 = zeros(n)
    Cq = zeros(k, n)
    G = want_gram ? zeros(n, n) : nothing
    _jacobian_blocks!(o.obj, θ) do rows, B, _
        a2 .+= vec(sum(abs2, B; dims = 1)) ./ ae.η^2
        mul!(Cq, view(ae.Q, rows, :)', B, 1 / ae.η, true)
        want_gram && _gram_update!(G, B, 1 / ae.η^2)
    end
    return a2, Cq, G                     # (G: upper triangle only)
end

function jacobian_column_norms(o::ApproximationErrorObjective, θ::AbstractVector)
    a2, Cq, _ = _whitened_blocks(o, θ, false)
    c = 2 .* o.ae.w .- o.ae.w .^ 2
    return sqrt.(max.(a2 .- vec(sum(c .* abs2.(Cq); dims = 1)), 0.0))
end

function jacobian_gram(o::ApproximationErrorObjective, θ::AbstractVector)
    _, Cq, G = _whitened_blocks(o, θ, true)
    c = 2 .* o.ae.w .- o.ae.w .^ 2
    G .-= Cq' * (c .* Cq)
    _symmetrize!(G)
    r = residual!(similar(o.r), o, θ)
    g = jacobian_operator(o, θ)' * r
    return G, g
end

_has_row_blocks(::ApproximationErrorObjective) = true
