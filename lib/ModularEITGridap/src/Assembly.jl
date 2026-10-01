# Matrices of a Gridap discretization and its conductivity tensor.

function FEMatrices(d::GridapDiscretization)
    dΩ = d.dΩ
    M_u = assemble_matrix((u, v) -> ∫(u * v)dΩ, d.U, d.V)
    K_u = assemble_matrix((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, d.U, d.V)
    M_Γ = _boundary_mass(d, d.boundary_facets)
    M_σ = assemble_matrix((u, v) -> ∫(u * v)dΩ, d.Uσ, d.Vσ)
    K_σ = assemble_matrix((u, v) -> ∫(∇(u) ⋅ ∇(v))dΩ, d.Uσ, d.Vσ)
    return FEMatrices(_csc(M_u), _csc(K_u), _csc(M_Γ), _csc(M_σ), _csc(K_σ), cholesky(Symmetric(_csc(M_σ))))
end

_csc(A) = SparseMatrixCSC{Float64, Int}(A)

function assemble_weighted_stiffness(d::GridapDiscretization, σ::AbstractVector)
    σh = FEFunction(d.Vσ, Vector{Float64}(σ))
    return _csc(assemble_matrix((u, v) -> ∫(σh * (∇(u) ⋅ ∇(v)))d.dΩ, d.U, d.V))
end

# Per cell, at the quadrature points of dΩ: physical gradients of the u basis (nq × nb), values of
# the σ basis (nq × nψ) and the weights × |det J| (nq). Gridap integrands combine at most a trial
# and a test basis, so the three-index tensor ∫ ψₐ ∇φᵢ⋅∇φⱼ is assembled from these values.
function _quadrature_data(d::GridapDiscretization)
    x = get_cell_points(d.dΩ)
    ∇φ = (∇(get_fe_basis(d.V)))(x)
    ψ = (get_fe_basis(d.Vσ))(x)
    quad = d.dΩ.quad
    Jt = lazy_map(∇, get_cell_map(quad.trian))
    Jx = lazy_map(evaluate, Jt, quad.cell_point)
    dV = lazy_map(Broadcasting((w, J) -> w * TensorValues.meas(J)), quad.cell_weight, Jx)
    return ∇φ, ψ, dV
end

function ConductivityTensor(d::GridapDiscretization; pattern = _u_pattern(d), to_device = identity)
    return ConductivityTensor(pattern, _conductivity_tensor(d, pattern); to_device)
end

# T (nnz(pattern) × n_σ) in COO form: for every cell the local tensor ∫ ψₐ ∇φᵢ⋅∇φⱼ, scattered to
# (nz index of (i, j), σ dof a); duplicates are summed by `sparse`.
function _conductivity_tensor(d::GridapDiscretization, pattern::SparseMatrixCSC)
    ∇φ, ψ, dV = _quadrature_data(d)
    c∇φ, cψ, cdV = array_cache(∇φ), array_cache(ψ), array_cache(dV)
    ids_u, ids_σ = get_cell_dof_ids(d.V), get_cell_dof_ids(d.Vσ)
    cu, cσ = array_cache(ids_u), array_cache(ids_σ)
    Is, Js, Vs = Int[], Int[], Float64[]
    for c in 1:num_cells(d.Ω)
        G = getindex!(c∇φ, ∇φ, c)
        P = getindex!(cψ, ψ, c)
        w = getindex!(cdV, dV, c)
        du = getindex!(cu, ids_u, c)
        dσ = getindex!(cσ, ids_σ, c)
        nq, nb = size(G)
        for (a, sa) in enumerate(dσ), j in 1:nb, i in 1:nb
            v = 0.0
            for q in 1:nq
                v += w[q] * P[q, a] * (G[q, i] ⋅ G[q, j])
            end
            push!(Is, _nz_index(pattern, du[i], du[j]))
            push!(Js, sa)
            push!(Vs, v)
        end
    end
    return sparse(Is, Js, Vs, nnz(pattern), ndofs_σ(d))
end
