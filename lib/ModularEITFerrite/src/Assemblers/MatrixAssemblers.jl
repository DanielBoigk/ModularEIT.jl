# Mass, stiffness, boundary mass and weighted stiffness matrices.
#
#   mass matrix               M = ∫ φᵢ φⱼ dΩ
#   stiffness matrix          K = ∫ ∇φᵢ⋅∇φⱼ dΩ
#   boundary mass matrix     M_Γ = ∫_Γ φᵢ φⱼ ds
#   weighted stiffness matrix L = ∫ σ ∇φᵢ⋅∇φⱼ dΩ   (σ from its own space)
#
# The `!` versions overwrite the values of a matrix with the sparsity pattern of
# `allocate_matrix(dh)` and allocate nothing else than one element matrix. For repeated
# assembly of L(σ) use the ConductivityTensor (one sparse matrix-vector product). Matrices are
# tied to one mesh; after adaptive refinement they are assembled again for the new
# discretization. The Dirichlet version of L is the principal submatrix on the free dofs, see
# `ForwardModel`.

function FEMatrices(disc::FerriteDiscretization)
    M_u = _condense(disc, assemble_mass(disc.dh_u, disc.cv_u))
    K_u = _condense(disc, assemble_stiffness(disc.dh_u, disc.cv_u))
    M_Γ = _condense(disc, assemble_boundary_mass(disc.dh_u, disc.fv_u, disc.boundary_facets))
    M_σ = assemble_mass(disc.dh_σ, disc.cv_σ)
    K_σ = assemble_stiffness(disc.dh_σ, disc.cv_σ)
    return FEMatrices(M_u, K_u, M_Γ, M_σ, K_σ, cholesky(Symmetric(M_σ)))
end

"""
    assemble_mass!(M, dh, cv)
    assemble_mass(dh, cv)

Mass matrix `∫ φᵢ φⱼ dΩ` of the (single-field) DofHandler `dh` with cell values `cv`.
"""
function assemble_mass!(M::SparseMatrixCSC, dh::DofHandler, cv::CellValues)
    return _assemble_bilinear!(M, dh, cv) do Ke, cv, q, dΩ
        n = getnbasefunctions(cv)
        for j in 1:n
            φⱼ = shape_value(cv, q, j)
            for i in 1:n
                Ke[i, j] += shape_value(cv, q, i) * φⱼ * dΩ
            end
        end
    end
end
assemble_mass(dh::DofHandler, cv::CellValues) = assemble_mass!(allocate_matrix(dh), dh, cv)

"""
    assemble_stiffness!(K, dh, cv)
    assemble_stiffness(dh, cv)

Stiffness matrix `∫ ∇φᵢ⋅∇φⱼ dΩ`.
"""
function assemble_stiffness!(K::SparseMatrixCSC, dh::DofHandler, cv::CellValues)
    return _assemble_bilinear!(K, dh, cv) do Ke, cv, q, dΩ
        n = getnbasefunctions(cv)
        for j in 1:n
            ∇φⱼ = shape_gradient(cv, q, j)
            for i in 1:n
                Ke[i, j] += (shape_gradient(cv, q, i) ⋅ ∇φⱼ) * dΩ
            end
        end
    end
end
assemble_stiffness(dh::DofHandler, cv::CellValues) = assemble_stiffness!(allocate_matrix(dh), dh, cv)

# element loop shared by the constant-coefficient bilinear forms
function _assemble_bilinear!(integrand!::F, A::SparseMatrixCSC, dh::DofHandler, cv::CellValues) where {F}
    n = getnbasefunctions(cv)
    Ke = zeros(n, n)
    assembler = start_assemble(A)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cv, cell)
        for q in 1:getnquadpoints(cv)
            integrand!(Ke, cv, q, getdetJdV(cv, q))
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return A
end

"""
    assemble_boundary_mass!(M, dh, fv, facets)
    assemble_boundary_mass(dh, fv, facets)

Boundary mass matrix `∫_Γ φᵢ φⱼ ds` over the facets `facets`, with facet values `fv`.
"""
function assemble_boundary_mass!(M::SparseMatrixCSC, dh::DofHandler, fv::FacetValues, facets)
    n = getnbasefunctions(fv)
    Me = zeros(n, n)
    assembler = start_assemble(M)
    for fc in FacetIterator(dh, Set(facets))
        fill!(Me, 0)
        reinit!(fv, fc)
        for q in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q)
            for j in 1:n
                φⱼ = shape_value(fv, q, j)
                for i in 1:n
                    Me[i, j] += shape_value(fv, q, i) * φⱼ * dΓ
                end
            end
        end
        assemble!(assembler, celldofs(fc), Me)
    end
    return M
end
assemble_boundary_mass(dh::DofHandler, fv::FacetValues, facets) =
    assemble_boundary_mass!(allocate_matrix(dh), dh, fv, facets)

"""
    assemble_boundary_load!(f, dh, fv, facets)

Boundary load vector `fᵢ = ∫_Γ φᵢ ds` over `facets` (added to `f`).
"""
function assemble_boundary_load!(f::AbstractVector, dh::DofHandler, fv::FacetValues, facets)
    n = getnbasefunctions(fv)
    fe = zeros(n)
    for fc in FacetIterator(dh, Set(facets))
        fill!(fe, 0)
        reinit!(fv, fc)
        for q in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q)
            for i in 1:n
                fe[i] += shape_value(fv, q, i) * dΓ
            end
        end
        assemble!(f, celldofs(fc), fe)
    end
    return f
end

"""
    assemble_weighted_stiffness!(L, disc, σ)
    assemble_weighted_stiffness(disc, σ)

Weighted stiffness matrix `∫ σ ∇φᵢ⋅∇φⱼ dΩ` by a classical element loop, with σ given by its
coefficients in the σ space of `disc`. The `!` version fills a matrix with the pattern of
`allocate_matrix(disc.dh_u)` (all dofs, before conformity constraints); the allocating version
returns the matrix of the u space of `disc` (condensed on non-conforming grids). For repeated
assembly on a fixed mesh, the [`ConductivityTensor`](@ref) method is a single sparse
matrix-vector product.
"""
function assemble_weighted_stiffness!(L::SparseMatrixCSC, disc::FerriteDiscretization, σ::AbstractVector)
    cv_u, cv_σ = disc.cv_u, disc.cv_σ
    nu, nσ = getnbasefunctions(cv_u), getnbasefunctions(cv_σ)
    Le = zeros(nu, nu)
    σdofs = zeros(Int, nσ)
    σe = zeros(eltype(σ), nσ)
    assembler = start_assemble(L)
    for cell in CellIterator(disc.dh_u)
        fill!(Le, 0)
        reinit!(cv_u, cell)
        reinit!(cv_σ, cell)
        celldofs!(σdofs, disc.dh_σ, cellid(cell))
        for (a, d) in enumerate(σdofs)
            σe[a] = σ[d]
        end
        for q in 1:getnquadpoints(cv_u)
            σq = function_value(cv_σ, q, σe)
            dΩ = getdetJdV(cv_u, q)
            for j in 1:nu
                ∇φⱼ = shape_gradient(cv_u, q, j)
                for i in 1:nu
                    Le[i, j] += σq * (shape_gradient(cv_u, q, i) ⋅ ∇φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Le)
    end
    return L
end
assemble_weighted_stiffness(disc::FerriteDiscretization, σ::AbstractVector) =
    _condense(disc, assemble_weighted_stiffness!(allocate_matrix(disc.dh_u), disc, σ))
