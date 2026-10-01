# Matrix-free Jacobians of parametrized objectives, and quantities accumulated from row blocks of
# the Jacobian (column norms, Gram matrix JᵀJ) without storing it.

"""
    ParametrizedJacobian

Matrix-free Jacobian `J_θ = J_σ P` of a [`ParametrizedObjective`](@ref); see
[`jacobian_operator`](@ref).
"""
struct ParametrizedJacobian{J, MP}
    Jσ::J
    P::MP
    buf::Vector{Float64}
end

jacobian_operator(o::ParametrizedObjective, θ::AbstractVector) =
    ParametrizedJacobian(jacobian_operator(o.obj, conductivity(o.par, θ)), o.par.P, zeros(size(o.par.P, 1)))

Base.size(J::ParametrizedJacobian) = (size(J.Jσ, 1), size(J.P, 2))
Base.size(J::ParametrizedJacobian, d::Integer) = d == 1 ? size(J.Jσ, 1) : d == 2 ? size(J.P, 2) : 1
Base.eltype(::ParametrizedJacobian) = Float64
Base.adjoint(J::ParametrizedJacobian) = AdjointJacobianOperator(J)
Base.:*(J::ParametrizedJacobian, v::AbstractVector) = mul!(zeros(size(J, 1)), J, v)

LinearAlgebra.mul!(y::AbstractVector, J::ParametrizedJacobian, v::AbstractVector) = mul!(y, J.Jσ, mul!(J.buf, J.P, v))
function LinearAlgebra.mul!(g::AbstractVector, A::AdjointJacobianOperator{<:ParametrizedJacobian}, w::AbstractVector)
    J = A.J
    mul!(J.buf, J.Jσ', w)
    return mul!(g, J.P', J.buf)
end

function _jacobian_blocks!(f, o::ParametrizedObjective, θ::AbstractVector)
    P = o.par.P
    buf = zeros(64, size(P, 2))                      # (row blocks have at most 64 rows)
    return _jacobian_blocks!(o.obj, conductivity(o.par, θ)) do rows, B, rr
        Bθ = view(buf, 1:size(B, 1), :)
        mul!(Bθ, B, P)
        f(rows, Bθ, rr)
    end
end

# whether an objective provides Jacobian row blocks (column norms, Gram matrix without J)
_has_row_blocks(obj) = hasmethod(_jacobian_blocks!, Tuple{Function, typeof(obj), Vector{Float64}})

"""
    jacobian_column_norms(obj, θ)

Column norms `‖J eⱼ‖` of the Jacobian of the least-squares objective `obj` at `θ`
(sensitivities), accumulated from row blocks without storing `J`.
"""
function jacobian_column_norms(obj::AbstractObjective, θ::AbstractVector)
    acc = zeros(length(θ))
    _jacobian_blocks!(obj, θ) do _, B, _
        acc .+= vec(sum(abs2, B; dims = 1))
    end
    return sqrt.(acc)
end

"""
    jacobian_gram(obj, θ)

The Gram matrix `JᵀJ` (`n × n`) and `Jᵀr` of the Jacobian and residual of `obj` at `θ`,
accumulated from row blocks without storing `J`: the Gauss–Newton Hessian and gradient, also
for problems with far more residuals than parameters.
"""
function jacobian_gram(obj::AbstractObjective, θ::AbstractVector)
    n = length(θ)
    G, g = zeros(n, n), zeros(n)
    _jacobian_blocks!(obj, θ) do _, B, rr
        _gram_update!(G, B, 1.0)
        mul!(g, B', rr, true, true)
    end
    return _symmetrize!(G), g
end

# G ← G + α BᵀB in the upper triangle only, mirrored once by _symmetrize! at the end. (mul!(G, B',
# B, α, true) calls syrk too, but mirrors the triangle of all of G after every update: for
# 16384 pixels 5–10 s per 64-row block, against 0.07 s for the update itself.)
_gram_update!(G::StridedMatrix{Float64}, B::StridedMatrix{Float64}, α::Real) = BLAS.syrk!('U', 'T', Float64(α), B, 1.0, G)
_gram_update!(G, B, α) = mul!(G, B', B, α, true)
_symmetrize!(G::StridedMatrix{Float64}) = LinearAlgebra.copytri!(G, 'U')
_symmetrize!(G) = G

"""
    residual(obj, θ)

The residual vector of a least-squares objective (`J = ½ ‖r‖²`) at `θ`, allocating; see
[`residual!`](@ref). For an [`AdjointStateObjective`](@ref) with zero data and the plain misfit
this is the vector of measured voltages, i.e. the forward map. With ChainRulesCore loaded, it has
reverse (`Jᵀ r̄`, adjoint solves) and forward (`J δθ`, linearized solves) differentiation rules,
as does [`objective_value`](@ref) (reverse: the adjoint-state gradient).
"""
residual(obj::AbstractObjective, θ::AbstractVector) = residual!(zeros(n_residual(obj)), obj, θ)
