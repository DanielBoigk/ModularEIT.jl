# Regularizers on a Ferrite discretization: Gram matrices of the σ space and the facet graph of
# piecewise constants.
#
# Piecewise constants have no gradient, so K_σ = 0 and the H¹ seminorm is not available. The
# `:jump` penalty is its two-point-flux counterpart (as in finite volume methods):
#     ½ Σ_F |F|/d_F (σ_K - σ_K')²,   d_F = distance of the centroids of K and K',
# which equals ½ ∫|∇σ|² for the interpolant of a linear function on orthogonal meshes (up to the
# half cells at the boundary).

# interior facets of a piecewise constant σ space: dofs of both cells, facet measure, centroid
# distance
struct _FacetGraph
    i::Vector{Int}
    j::Vector{Int}
    len::Vector{Float64}
    dist::Vector{Float64}
end

_is_piecewise_constant(d::FerriteDiscretization) =
    d.ip_σ isa DiscontinuousLagrange && Ferrite.getorder(d.ip_σ) == 0

function _facet_graph(d::FerriteDiscretization)
    _is_piecewise_constant(d) || throw(ArgumentError("the facet graph needs piecewise constant σ"))
    grid = d.grid
    centroid(c) = (nodes = getcells(grid, c).nodes; sum(n -> get_node_coordinate(grid, n), nodes) / length(nodes))
    m = length(d.interior_facets)
    fg = _FacetGraph(zeros(Int, m), zeros(Int, m), zeros(m), zeros(m))
    for (k, (c1, f1, c2)) in enumerate(d.interior_facets)
        fg.i[k] = only(celldofs(d.dh_σ, c1))
        fg.j[k] = only(celldofs(d.dh_σ, c2))
        fg.len[k] = _facet_measure(d, FacetIndex(c1, f1))
        fg.dist[k] = norm(centroid(c1) - centroid(c2))
    end
    return fg
end

# weighted graph Laplacian Σ_k w_k (e_i - e_j)(e_i - e_j)ᵀ
function _graph_laplacian(fg::_FacetGraph, w::AbstractVector, n::Integer)
    I = vcat(fg.i, fg.j, fg.i, fg.j)
    J = vcat(fg.i, fg.j, fg.j, fg.i)
    V = vcat(w, w, -w, -w)
    return sparse(I, J, V, n, n)
end

"""
    TikhonovRegularizer(disc; kind = :L2, reference = 0, mats = nothing)

Tikhonov regularizer `½ ‖σ - σ₀‖²` on the σ space of `disc`:

- `:L2`: `½ ∫ (σ - σ₀)²` (mass matrix `M_σ`),
- `:H1semi`: `½ ∫ |∇(σ - σ₀)|²` (stiffness matrix `K_σ`, continuous σ only),
- `:H1`: sum of both (continuous σ only),
- `:jump`: `½ Σ_F |F|/d_F (σ_K - σ_K')²` over interior facets (piecewise constant σ only), the
  two-point-flux version of the H¹ seminorm with centroid distances `d_F`.

Pass `mats = FEMatrices(disc)` to reuse assembled matrices.
"""
function TikhonovRegularizer(d::FerriteDiscretization; kind::Symbol = :L2, reference = 0.0, mats = nothing)
    kind in (:L2, :H1semi, :H1, :jump) ||
        throw(ArgumentError("kind must be :L2, :H1semi, :H1 or :jump, got :$kind"))
    pc = _is_piecewise_constant(d)
    if kind === :jump
        pc || throw(ArgumentError("the :jump penalty is for piecewise constant σ; use :H1semi for continuous σ"))
        fg = _facet_graph(d)
        G = _graph_laplacian(fg, fg.len ./ fg.dist, ndofs_σ(d))
    else
        kind !== :L2 && pc &&
            throw(ArgumentError("piecewise constant σ has no H¹ seminorm (K_σ = 0); use kind = :jump"))
        G = _gram_matrix(d, :σ, kind, mats)
    end
    return TikhonovRegularizer(G; reference)
end

function TotalVariationRegularizer(d::FerriteDiscretization; ε::Real = 1e-3)
    ε >= 0 || throw(ArgumentError("ε must be nonnegative"))
    cache = _is_piecewise_constant(d) ? _facet_graph(d) : nothing
    return TotalVariationRegularizer(d, Float64(ε), cache)
end

objective_value(reg::TotalVariationRegularizer{<:FerriteDiscretization}, σ::AbstractVector) = _tv!(nothing, reg, σ)
value_and_gradient!(g::AbstractVector, reg::TotalVariationRegularizer{<:FerriteDiscretization}, σ::AbstractVector) =
    _tv!(g, reg, σ)

_tv!(g, reg::TotalVariationRegularizer{<:FerriteDiscretization, Nothing}, σ) = _total_variation!(g, reg.disc, σ, reg.ε)

function _tv!(g, reg::TotalVariationRegularizer{<:FerriteDiscretization, _FacetGraph}, σ)
    fg, ε = reg.cache, reg.ε
    g === nothing || fill!(g, 0)
    tv = zero(eltype(σ))
    @inbounds for k in eachindex(fg.i)
        i, j = fg.i[k], fg.j[k]
        jump = σ[i] - σ[j]
        s = sqrt(jump^2 + ε^2)
        tv += fg.len[k] * s
        if g !== nothing && s > 0
            dj = fg.len[k] * jump / s
            g[i] += dj
            g[j] -= dj
        end
    end
    return tv
end

# lagged diffusivity: the quadratic form with weights frozen at σ, so that ∇TV(σ) = H(σ) σ
function gauss_newton_hessian(reg::TotalVariationRegularizer{<:FerriteDiscretization, _FacetGraph}, σ::AbstractVector)
    fg, ε = reg.cache, reg.ε
    w = similar(fg.len)
    for k in eachindex(w)
        s = sqrt((σ[fg.i[k]] - σ[fg.j[k]])^2 + ε^2)
        w[k] = s > 0 ? fg.len[k] / s : 0.0
    end
    return _graph_laplacian(fg, w, length(σ))
end

function gauss_newton_hessian(reg::TotalVariationRegularizer{<:FerriteDiscretization, Nothing}, σ::AbstractVector)
    d, ε = reg.disc, reg.ε
    cv = d.cv_σ
    n = getnbasefunctions(cv)
    σe = zeros(eltype(σ), n)
    He = zeros(n, n)
    H = allocate_matrix(d.dh_σ)
    assembler = start_assemble(H)
    for cell in CellIterator(d.dh_σ)
        reinit!(cv, cell)
        dofs = celldofs(cell)
        for (a, k) in enumerate(dofs)
            σe[a] = σ[k]
        end
        fill!(He, 0)
        for q in 1:getnquadpoints(cv)
            ∇σ = function_gradient(cv, q, σe)
            s = sqrt(∇σ ⋅ ∇σ + ε^2)
            s > 0 || continue
            w = getdetJdV(cv, q) / s
            for b in 1:n
                ∇φb = shape_gradient(cv, q, b)
                for a in 1:n
                    He[a, b] += (shape_gradient(cv, q, a) ⋅ ∇φb) * w
                end
            end
        end
        assemble!(assembler, dofs, He)
    end
    return H
end
