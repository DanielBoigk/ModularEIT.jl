# Regularizer primitives of the Gridap back end (the regularizers are in ModularEIT,
# Galerkin/RegularizersFE.jl): the facet graph of piecewise constants, and the quadrature-based
# total variation pieces of continuous σ.

_is_piecewise_constant(d::GridapDiscretization) = d.order_σ == 0

function _facet_graph(d::GridapDiscretization)
    _is_piecewise_constant(d) || throw(ArgumentError("the facet graph needs piecewise constant σ"))
    D = num_cell_dims(d.model)
    topo = get_grid_topology(d.model)
    facet_cells = get_faces(topo, D - 1, D)
    ids = get_cell_dof_ids(d.Vσ)
    x = get_node_coordinates(d.model)
    cell_nodes = get_cell_node_ids(d.model)
    centroid(c) = sum(x[n] for n in cell_nodes[c]) / length(cell_nodes[c])
    i, j, len, dist = Int[], Int[], Float64[], Float64[]
    for f in eachindex(facet_cells)
        cells = facet_cells[f]
        length(cells) == 2 || continue
        c1, c2 = cells
        push!(i, only(ids[c1]))
        push!(j, only(ids[c2]))
        push!(len, _facet_measure(d, f))
        push!(dist, norm(centroid(c1) - centroid(c2)))
    end
    return _FacetGraph(i, j, len, dist)
end

total_variation(d::GridapDiscretization, σ::AbstractVector; ε = 0.0) = _total_variation!(nothing, d, σ, ε)
total_variation!(g::AbstractVector, d::GridapDiscretization, σ::AbstractVector; ε = 0.0) =
    _total_variation!(g, d, σ, ε)

function _total_variation!(g, d::GridapDiscretization, σ, ε)
    g === nothing || fill!(g, 0)
    if _is_piecewise_constant(d)
        fg = _facet_graph(d)
        tv = zero(eltype(σ))
        for k in eachindex(fg.i)
            i, j = fg.i[k], fg.j[k]
            jump = σ[i] - σ[j]
            s = sqrt(jump^2 + ε^2)
            tv += fg.len[k] * s
            if g !== nothing && s > 0
                g[i] += fg.len[k] * jump / s
                g[j] -= fg.len[k] * jump / s
            end
        end
        return tv
    end
    σh = FEFunction(d.Vσ, Vector{Float64}(σ))
    s = sqrt ∘ (∇(σh) ⋅ ∇(σh) + ε^2)
    tv = sum(∫(s)d.dΩ)
    g === nothing || copyto!(g, assemble_vector(v -> ∫((∇(σh) ⋅ ∇(v)) / s)d.dΩ, d.Vσ))
    return tv
end

function _tv_hessian(d::GridapDiscretization, σ::AbstractVector, ε)
    σh = FEFunction(d.Vσ, Vector{Float64}(σ))
    s = sqrt ∘ (∇(σh) ⋅ ∇(σh) + ε^2)
    return _csc(assemble_matrix((u, v) -> ∫((∇(u) ⋅ ∇(v)) / s)d.dΩ, d.Uσ, d.Vσ))
end

# gradients of continuous σ at the quadrature points (rows, `dim` per point) and the weights
function _tv_gradient_operator(d::GridapDiscretization)
    x = get_cell_points(d.dΩ)
    ∇ψ = (∇(get_fe_basis(d.Vσ)))(x)
    _, _, dV = _quadrature_data(d)
    dim = _spatial_dim(d)
    ids = get_cell_dof_ids(d.Vσ)
    I, J, V, w = Int[], Int[], Float64[], Float64[]
    row = 0
    for c in 1:num_cells(d.Ω)
        G, wq, dofs = ∇ψ[c], dV[c], ids[c]
        for q in axes(G, 1)
            push!(w, wq[q])
            for (a, k) in enumerate(dofs), comp in 1:dim
                push!(I, row + comp); push!(J, k); push!(V, G[q, a][comp])
            end
            row += dim
        end
    end
    return sparse(I, J, V, row, ndofs_σ(d)), w, dim
end
