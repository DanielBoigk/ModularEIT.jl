# Functions on the finite element spaces: interpolation, L² projection, inner products, norms.

function interpolate_function(d::GridapDiscretization, f; field::Symbol = :σ)
    _, space = _space(d, field)
    return Vector{Float64}(get_free_dof_values(interpolate(x -> f(x), space)))
end

"""
    l2_project(disc::GridapDiscretization, f; field = :σ, mats = nothing, quadrature_order = nothing)

L² projection of the function `f(x)` onto the σ or u space. The right-hand side is integrated
exactly for polynomials of degree `quadrature_order` (default: that of the discretization plus 2).
"""
function l2_project(d::GridapDiscretization, f; field::Symbol = :σ, mats = nothing, quadrature_order = nothing)
    trial, test = _space(d, field)
    dΩ = Measure(d.Ω, quadrature_order === nothing ? d.degree + 2 : Int(quadrature_order))
    g = x -> f(x)
    b = assemble_vector(v -> ∫(v * g)dΩ, test)
    M = mats === nothing ? _csc(assemble_matrix((u, v) -> ∫(u * v)d.dΩ, trial, test)) :
        field === :σ ? mats.M_σ : mats.M_u
    return Vector{Float64}(cholesky(Symmetric(M)) \ b)
end

function _gram_matrix(d::GridapDiscretization, field::Symbol, kind::Symbol, mats)
    kind in (:L2, :H1semi, :H1) || throw(ArgumentError("kind must be :L2, :H1semi or :H1, got :$kind"))
    if mats === nothing
        trial, test = _space(d, field)
        M = kind === :H1semi ? nothing : _csc(assemble_matrix((u, v) -> ∫(u * v)d.dΩ, trial, test))
        K = kind === :L2 ? nothing : _csc(assemble_matrix((u, v) -> ∫(∇(u) ⋅ ∇(v))d.dΩ, trial, test))
    else
        M = field === :σ ? mats.M_σ : mats.M_u
        K = field === :σ ? mats.K_σ : mats.K_u
    end
    return kind === :L2 ? M : kind === :H1semi ? K : M + K
end

function fe_inner(d::GridapDiscretization, a::AbstractVector, b::AbstractVector;
                  field::Symbol = :σ, kind::Symbol = :L2, mats = nothing)
    return dot(a, _gram_matrix(d, field, kind, mats), b)
end

fe_norm(d::GridapDiscretization, a::AbstractVector; kwargs...) = sqrt(max(fe_inner(d, a, a; kwargs...), 0))

lumped_mass(d::GridapDiscretization) = vec(sum(_gram_matrix(d, :σ, :L2, nothing); dims = 2))
