# Functionals on the finite element spaces: inner products, norms and total variation.

"""
    fe_inner(disc, a, b; field = :σ, kind = :L2, mats = nothing)

Inner product of the finite element functions with coefficients `a`, `b` in the `:u` or `:σ`
space. `kind`: `:L2` (`∫ a b`), `:H1semi` (`∫ ∇a⋅∇b`) or `:H1` (sum of both). Pass
`mats = FEMatrices(disc)` to reuse assembled matrices.
"""
function fe_inner(d::FerriteDiscretization, a::AbstractVector, b::AbstractVector;
                  field::Symbol = :σ, kind::Symbol = :L2, mats = nothing)
    G = _gram_matrix(d, field, kind, mats)
    return dot(a, G, b)
end

"""
    fe_norm(disc, a; field = :σ, kind = :L2, mats = nothing)

Norm induced by [`fe_inner`](@ref).
"""
fe_norm(d::FerriteDiscretization, a::AbstractVector; kwargs...) = sqrt(max(fe_inner(d, a, a; kwargs...), 0))

function _gram_matrix(d::FerriteDiscretization, field::Symbol, kind::Symbol, mats)
    kind in (:L2, :H1semi, :H1) || throw(ArgumentError("kind must be :L2, :H1semi or :H1, got :$kind"))
    if mats === nothing
        dh, cv = _field_dh(d, field), _field_cv(d, field)
        cond(A) = field === :u ? _condense(d, A) : A
        M = kind === :H1semi ? nothing : cond(assemble_mass(dh, cv))
        K = kind === :L2 ? nothing : cond(assemble_stiffness(dh, cv))
    else
        M = field === :σ ? mats.M_σ : mats.M_u
        K = field === :σ ? mats.K_σ : mats.K_u
    end
    return kind === :L2 ? M : kind === :H1semi ? K : M + K
end

"""
    total_variation(disc, σ; ε = 0)
    total_variation!(g, disc, σ; ε = 0)

(Smoothed) total variation of the conductivity `σ`; the `!` version also writes the gradient
with respect to the coefficients into `g`.

- Piecewise constant σ (`DiscontinuousLagrange` of order 0): `Σ_F |F| √((σ_K - σ_K')² + ε²)`
  over the interior facets `F` between cells `K`, `K'`, the exact total variation for `ε = 0`.
- Continuous σ: `∫ √(|∇σ|² + ε²) dx`.

`ε > 0` makes the functional differentiable (see the wiki article on smoothed total variation).
"""
total_variation(d::FerriteDiscretization, σ::AbstractVector; ε = 0.0) = _total_variation!(nothing, d, σ, ε)
total_variation!(g::AbstractVector, d::FerriteDiscretization, σ::AbstractVector; ε = 0.0) =
    _total_variation!(g, d, σ, ε)

function _total_variation!(g, d::FerriteDiscretization, σ, ε)
    g === nothing || fill!(g, 0)
    if d.ip_σ isa DiscontinuousLagrange
        Ferrite.getorder(d.ip_σ) == 0 ||
            throw(ArgumentError("total variation of discontinuous σ is implemented for piecewise constants only"))
        return _tv_jumps!(g, d, σ, ε)
    end
    return _tv_gradient!(g, d, σ, ε)
end

# piecewise constants: facet jumps. The P0 dof of a cell is its only cell dof.
function _tv_jumps!(g, d::FerriteDiscretization, σ, ε)
    tv = zero(eltype(σ))
    for (c1, f1, c2) in d.interior_facets
        i, j = only(celldofs(d.dh_σ, c1)), only(celldofs(d.dh_σ, c2))
        len = _facet_measure(d, FacetIndex(c1, f1))
        jump = σ[i] - σ[j]
        s = sqrt(jump^2 + ε^2)
        tv += len * s
        if g !== nothing && s > 0
            dj = len * jump / s
            g[i] += dj
            g[j] -= dj
        end
    end
    return tv
end

# continuous σ: ∫ √(|∇σ|² + ε²)
function _tv_gradient!(g, d::FerriteDiscretization, σ, ε)
    cv = d.cv_σ
    n = getnbasefunctions(cv)
    σe = zeros(eltype(σ), n)
    ge = zeros(eltype(σ), n)
    tv = zero(eltype(σ))
    for cell in CellIterator(d.dh_σ)
        reinit!(cv, cell)
        dofs = celldofs(cell)
        for (a, k) in enumerate(dofs)
            σe[a] = σ[k]
        end
        fill!(ge, 0)
        for q in 1:getnquadpoints(cv)
            ∇σ = function_gradient(cv, q, σe)
            s = sqrt(∇σ ⋅ ∇σ + ε^2)
            dΩ = getdetJdV(cv, q)
            tv += s * dΩ
            if g !== nothing && s > 0
                for a in 1:n
                    ge[a] += (∇σ ⋅ shape_gradient(cv, q, a)) / s * dΩ
                end
            end
        end
        g === nothing || assemble!(g, dofs, ge)
    end
    return tv
end
