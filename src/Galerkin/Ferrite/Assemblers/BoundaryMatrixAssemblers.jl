using LinearAlgebra, SparseArrays, Ferrite

export assemble_boundary_M_K

function assemble_boundary_M_K(facetvalues::FacetValues, dh::DofHandler, ∂Ω, b_dofs)
    ndof = ndofs(dh)
    M = allocate_matrix(fe.dh)
    K = allocate_matrix(fe.dh)
    n_basefuncs = getnbasefunctions(facetvalues)
    Me = zeros(n_basefuncs, n_basefuncs)
    Ke = zeros(n_basefuncs, n_basefuncs)

    for facet in FacetIterator(dh, ∂Ω)
        fill!(Me, 0.0); fill!(Ke, 0.0)
        reinit!(facetvalues, facet)
        dofs = celldofs(facet)
        for q in 1:getnquadpoints(facetvalues)
            dΓ = getdetJdV(facetvalues, q)
            for i in 1:n_basefuncs
                φᵢ  = shape_value(facetvalues, q, i)
                ∇φᵢ = shape_gradient(facetvalues, q, i)  # tangential grad on the facet manifold
                for j in 1:n_basefuncs
                    φⱼ  = shape_value(facetvalues, q, j)
                    ∇φⱼ = shape_gradient(facetvalues, q, j)
                    Me[i, j] += φᵢ * φⱼ * dΓ
                    Ke[i, j] += (∇φᵢ ⋅ ∇φⱼ) * dΓ
                end
            end
        end
        for (a, A) in enumerate(dofs), (b, B) in enumerate(dofs)
            M[A, B] += Me[a, b]
            K[A, B] += Ke[a, b]
        end
    end
    return M[b_dofs, b_dofs], K[b_dofs, b_dofs]
end

function assemble_boundary_M_K(fe::FerriteFESpace)
    assemble_boundary_M_K(fe.facetvalues, fe.dh, fe.∂Ω, fe.b_dofs)
end

export get_i_dofs
function get_i_dofs(b_dofs, n::Integer)
    is_boundary = falses(n)
    is_boundary[b_dofs] .= true
    return findall(!, is_boundary)
end

function get_i_dofs(fe::FerriteFESpace) # return dofs for interior points
    get_i_dofs(fe.b_dofs, fe.n)
end


export assemble_fractional_M 
function assemble_fractional_M(M_Γ, K_Γ)
    Mb = M_Γ |> Matrix |> Symmetric
    Kb = K_Γ |> Matrix |> Symmetric
    λ, Φ = eigen(Kb, Mb) 
    λ = 
    A(s) = Mb * Φ * Diagonal((1.0 .+ λ).^s) * Φ' * Mb |> Symmetric
    Hn½ = A(-0.5)
    H½ = A(0.5)
    return Hn½, H½
end

export assemble_boundary_matrices
function assemble_boundary_matrices(fe::FerriteFESpace)
    MΓ, KΓ =  assemble_boundary_M_K(fe::FerriteFESpace)
    Hn½, H½ = assemble_fractional_M(M_Γ, K_Γ)
    fe.MΓ, fe.KΓ, fe.Hn½, fe.H½ = MΓ, KΓ, Hn½, H½
    return MΓ, KΓ, Hn½, H½
end 