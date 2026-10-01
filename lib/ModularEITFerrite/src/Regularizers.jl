# Regularizer primitives of the Ferrite back end (the regularizers are in ModularEIT,
# Galerkin/RegularizersFE.jl): the facet graph of piecewise constants, and the quadrature-based
# total variation pieces of continuous σ.
#
# Piecewise constants have no gradient, so K_σ = 0 and the H¹ seminorm is not available. The
# `:jump` penalty is its two-point-flux counterpart (as in finite volume methods):
#     ½ Σ_F |F|/d_F (σ_K - σ_K')²,   d_F = distance of the centroids of K and K',
# which equals ½ ∫|∇σ|² for the interpolant of a linear function on orthogonal meshes (up to the
# half cells at the boundary).

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

function _tv_hessian(d::FerriteDiscretization, σ::AbstractVector, ε)
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

# gradients of continuous σ at the quadrature points (rows) and the quadrature weights
function _tv_gradient_operator(d::FerriteDiscretization)
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
    return sparse(I, J, V, row, ndofs_σ(d)), w, dim
end


"""
    lumped_mass(disc)

Row sums of the σ mass matrix (the diagonal of `M_σ` for piecewise constants): the diagonal
metric of the discrete L² inner product used by [`prox!`](@ref), [`ProximalGradient`](@ref)
and [`ADMM`](@ref).
"""
lumped_mass(d::FerriteDiscretization) = vec(sum(assemble_mass(d.dh_σ, d.cv_σ); dims = 2))

