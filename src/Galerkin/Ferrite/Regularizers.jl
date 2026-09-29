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
    return TotalVariationRegularizer(d, Float64(ε), cache, Ref{Any}(nothing))
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

"""
    lumped_mass(disc)

Row sums of the σ mass matrix (the diagonal of `M_σ` for piecewise constants): the diagonal
metric of the discrete L² inner product used by [`prox!`](@ref), [`ProximalGradient`](@ref)
and [`ADMM`](@ref).
"""
lumped_mass(d::FerriteDiscretization) = vec(sum(assemble_mass(d.dh_σ, d.cv_σ); dims = 2))

# ---------------------------------------------------------------------------------------------
# Exact prox of the (non-smooth) total variation
#
#     min_{lo ≤ z ≤ hi}  Σ_g w_g ‖(K z)_g‖₂ + ρ/2 Σᵢ mᵢ (zᵢ - vᵢ)²
#
# K: facet differences (piecewise constants, groups of size 1, w_g = |F|) or gradients at the
# quadrature points (continuous σ, groups of size dim, w_g = quadrature weight × |det J|).
# Substituting y = D z, D = diag(√(ρm)), makes the data term ½‖y - D v‖² (strongly convex with
# modulus 1) and the operator K D⁻¹ with scalar steps τ = σ = 1/‖K D⁻¹‖ (Chambolle–Pock,
# Algorithm 1). The accelerated variant (Algorithm 2, τₖ → 0) converges only like O(1/k) in
# the iterates, noticeably so from warm starts, while the fixed-step iteration converges
# linearly on this polyhedral + quadratic problem. Stopping on the primal–dual gap (closed form
# below), which bounds ‖y - y*‖² ≤ 2 gap. The dual variable is kept between calls (warm start
# inside ADMM).

mutable struct _TVOperator
    K::SparseMatrixCSC{Float64, Int}
    Kt::SparseMatrixCSC{Float64, Int}
    w::Vector{Float64}            # group weights
    gd::Int                       # group size
    p::Vector{Float64}            # dual variable (warm start)
    m::Vector{Float64}            # metric of the cached norm
    L2::Float64                   # ‖K diag(1/√m)‖² (for ρ = 1)
end

function _tv_operator(d::FerriteDiscretization)
    if _is_piecewise_constant(d)
        fg = _facet_graph(d)
        k = length(fg.i)
        K = sparse(vcat(1:k, 1:k), vcat(fg.i, fg.j), vcat(ones(k), -ones(k)), k, ndofs_σ(d))
        return _TVOperator(K, sparse(K'), copy(fg.len), 1, zeros(k), Float64[], NaN)
    end
    cv = d.cv_σ
    dim = Ferrite.getspatialdim(d.grid)
    I, J, V, w = Int[], Int[], Float64[], Float64[]
    row = 0
    for cell in CellIterator(d.dh_σ)
        reinit!(cv, cell)
        dofs = celldofs(cell)
        for q in 1:getnquadpoints(cv)
            push!(w, getdetJdV(cv, q))
            for a in eachindex(dofs)
                ∇φ = shape_gradient(cv, q, a)
                for c in 1:dim
                    push!(I, row + c); push!(J, dofs[a]); push!(V, ∇φ[c])
                end
            end
            row += dim
        end
    end
    K = sparse(I, J, V, row, ndofs_σ(d))
    return _TVOperator(K, sparse(K'), w, dim, zeros(row), Float64[], NaN)
end

function _operator_norm2!(op::_TVOperator, m)
    op.m == m && return op.L2
    s = 1 ./ sqrt.(m)
    x = s .* (1 .+ 0.1 .* sin.(1:length(m)))           # deterministic start vector
    λ = 0.0
    for _ in 1:100
        y = s .* (op.Kt * (op.K * (s .* x)))
        λnew = norm(y) / norm(x)
        x = y ./ norm(y)
        abs(λnew - λ) <= 1e-6 * λnew && (λ = λnew; break)
        λ = λnew
    end
    op.m, op.L2 = copy(m), 1.05 * λ
    return op.L2
end

function prox!(z::AbstractVector, reg::TotalVariationRegularizer{<:FerriteDiscretization}, v::AbstractVector, ρ::Real;
               weights = nothing, lower = nothing, upper = nothing, tol = nothing, maxiter = nothing)
    reg.ε > 0 && return invoke(prox!, Tuple{AbstractVector, AbstractRegularizer, AbstractVector, Real}, z, reg, v, ρ;
                               weights, lower, upper)
    ρ > 0 || throw(ArgumentError("ρ must be positive"))
    reg.prox_state[] === nothing && (reg.prox_state[] = _tv_operator(reg.disc))
    op = reg.prox_state[]::_TVOperator
    n = length(v)
    m = weights === nothing ? ones(n) : Vector{Float64}(weights)
    lo = lower === nothing ? fill(-Inf, n) : lower isa Real ? fill(Float64(lower), n) : lower
    hi = upper === nothing ? fill(Inf, n) : upper isa Real ? fill(Float64(upper), n) : upper
    return _tv_prox_cp!(z, op, v, Float64(ρ), m, lo, hi; maxiter = something(maxiter, 100_000), gap_tol = tol)
end

# primal–dual gap of the scaled problem: P(y) - D(p), with the dual value
# D(p) = min_{y ∈ box} ⟨y, K̃ᵀp⟩ + ½‖y - Dv‖², attained at y* = clamp(Dv - K̃ᵀp)
function _tv_gap(op::_TVOperator, y, p, D, Dv, ylo, yhi, Ky, Ktp)
    mul!(Ky, op.K, y ./ D)
    P = sum(abs2, y .- Dv) / 2
    @inbounds for g in eachindex(op.w)
        r = (g - 1) * op.gd
        P += op.w[g] * sqrt(sum(c -> Ky[r + c]^2, 1:op.gd))
    end
    mul!(Ktp, op.Kt, p)
    q = Ktp ./ D
    ys = clamp.(Dv .- q, ylo, yhi)
    return P - (dot(ys, q) + sum(abs2, ys .- Dv) / 2), P
end

# Stops when the gap is below `gap_tol` (absolute; it bounds ρ/2 ‖z - z*‖²_m) or, by default, at
# the round-off level 1e-14 (1 + |P|). Callers that use the prox inexactly (ADMM, proximal
# gradient) pass tolerances tied to their own progress: the fixed-step iteration converges only
# slowly on meshes with very small or thin cells (e.g. near the centre of polar meshes).
function _tv_prox_cp!(z, op::_TVOperator, v, ρ, m, lo, hi; maxiter = 100_000, gap_tol = nothing)
    D = sqrt.(ρ .* m)
    Dv = D .* v
    ylo, yhi = D .* lo, D .* hi
    L = sqrt(_operator_norm2!(op, m) / ρ)                 # ‖K D⁻¹‖
    τ = σs = L > 0 ? 1 / L : 1.0
    y = clamp.(Dv, ylo, yhi)
    ybar = copy(y)
    yold = similar(y)
    p = op.p
    Ky = zeros(size(op.K, 1))
    Ktp = zeros(length(v))
    gd, w = op.gd, op.w
    for it in 1:maxiter
        # dual step: p ← proj_{‖p_g‖ ≤ w_g}(p + σ K D⁻¹ ȳ)
        mul!(Ky, op.K, ybar ./ D)
        p .+= σs .* Ky
        @inbounds for g in eachindex(w)
            r = (g - 1) * gd
            s = 0.0
            for c in 1:gd
                s += p[r + c]^2
            end
            s = sqrt(s)
            if s > w[g]
                f = w[g] / s
                for c in 1:gd
                    p[r + c] *= f
                end
            end
        end
        # primal step: y ← clamp((y - τ D⁻¹Kᵀp + τ D v) / (1 + τ))
        mul!(Ktp, op.Kt, p)
        copyto!(yold, y)
        @. y = clamp((y - τ * Ktp / D + τ * Dv) / (1 + τ), ylo, yhi)
        @. ybar = 2y - yold
        if it == 2 || it % 20 == 0                        # early check: warm starts may already suffice
            gap, P = _tv_gap(op, y, p, D, Dv, ylo, yhi, Ky, Ktp)
            gap <= max(something(gap_tol, 0.0), 1e-14 * (1 + abs(P))) && break
        end
    end
    z .= y ./ D
    return z
end
