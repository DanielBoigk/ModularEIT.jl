
using Ferrite
using SparseArrays

export calculate_bilinear_map!, calculate_bilinear_map
# Assemble the projection of ∇(u) ⋅ ∇(λ) onto the FE space.
# This computes rhs_i = ∫ (∇u ⋅ ∇λ) ϕ_i dΩ for each test function ϕ_i.
function calculate_bilinear_map!(fe::FerriteFESpace, rhs::AbstractVector, a::AbstractVector, b::AbstractVector)
    cellvalues = fe.cellvalues
    dh = fe.dh
    M = fe.M_fac
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    re = zeros(n_basefuncs)

    fill!(rhs, 0.0)
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(re, 0.0)
        ae = a[dofs]
        be = b[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇a_q = zero(Vec{2,Float64})
            ∇b_q = zero(Vec{2,Float64})
            for j in 1:n_basefuncs
                ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                ∇a_q += ae[j] * ∇ϕⱼ
                ∇b_q += be[j] * ∇ϕⱼ
            end
            grad_dot_product = ∇a_q ⋅ ∇b_q
            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                re[i] += grad_dot_product * ϕᵢ * dΩ
            end
        end
        assemble!(rhs, dofs, re)
    end
    return M \ rhs
end

function calculate_bilinear_map(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    rhs = zeros(fe.n)
    return calculate_bilinear_map!(fe, rhs, a, b)
end

#=
function calculate_discrete_gradient!(fe::FerriteFESpace, grad_σ::AbstractVector ,a::AbstractVector, b::AbstractVector)
    cellvalues = fe.cellvalues
    dh = fe.dh
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    
    # Pre-allocate local elemental stiffness matrix
    Ke0 = zeros(n_basefuncs, n_basefuncs)
    
    # The output is one gradient component per element/cell
    n_cells = fe.n
    fill!(grad_σ, 0.0)

    for (cell_idx, cell) in enumerate(CellIterator(dh))
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(Ke0, 0.0)
        
        # Local state and adjoint degrees of freedom
        ae = a[dofs]
        be = b[dofs]
        
        # 1. Standard local stiffness matrix assembly (with nominal σ = 1)
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                    # Standard Laplacian weak form integrand
                    Ke0[i, j] += (∇ϕᵢ ⋅ ∇ϕⱼ) * dΩ
                end
            end
        end
        
        # 2. Algebraic contraction: grad_i = a_e^T * Ke0 * b_e
        # In Julia, this quadratic form can be neatly written as:
        grad_σ[cell_idx] = ae ⋅ (Ke0 * be)
    end
    
    # Note: If your sensitivity definition requires a negative sign (e.g., -λ^T * dK/dσ * u), 
    # just return -grad_σ depending on how you've signed your adjoint vector.
    return grad_σ
end
=#
function calculate_discrete_gradient!(fe::FerriteFESpace, grad_σ::AbstractVector, a::AbstractVector, b::AbstractVector)
    cellvalues = fe.cellvalues
    dh = fe.dh
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)

    fill!(grad_σ, 0.0)

    for (cell_idx, cell) in enumerate(CellIterator(dh))
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)

        ae = a[dofs]
        be = b[dofs]

        local_sum = 0.0
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇a_q = zero(Vec{2,Float64})
            ∇b_q = zero(Vec{2,Float64})
            for j in 1:n_basefuncs
                ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                ∇a_q += ae[j] * ∇ϕⱼ
                ∇b_q += be[j] * ∇ϕⱼ
            end
            local_sum += (∇a_q ⋅ ∇b_q) * dΩ
        end

        grad_σ[cell_idx] = local_sum
    end

    return copy(grad_σ)
end

function calculate_discrete_gradient(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    grad_σ = zeros(fe.n)
    return calculate_discrete_gradient!(fe, grad_σ, a, b)  # was missing grad_σ and return
end
#=
function calculate_discrete_gradient!(fe::FerriteFESpace, grad_σ::AbstractVector, a::AbstractVector, b::AbstractVector)
    cellvalues = fe.cellvalues
    dh = fe.dh
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    
    # Ke_scaled is the element stiffness matrix for a unit conductivity (σ = 1)
    Ke_scaled = zeros(n_basefuncs, n_basefuncs)

    fill!(grad_σ, 0.0)
    
    # Loop over every element (assuming grad_σ is indexed by cell/element ID)
    for (cell_idx, cell) in enumerate(CellIterator(dh))
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(Ke_scaled, 0.0)
        
        # 1. Assemble the standard local element stiffness matrix (for σ = 1)
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                    Ke_scaled[i, j] += (∇ϕᵢ ⋅ ∇ϕⱼ) * dΩ
                end
            end
        end
        
        # 2. Extract local nodal values for forward (a) and adjoint (b) solutions
        ae = a[dofs]
        be = b[dofs]
        
        # 3. Compute the gradient contribution for this element: -aeᵀ * Ke_scaled * be
        # (The negative sign comes from the adjoint derivation)
        grad_σ[cell_idx] = -dot(ae, Ke_scaled * be)
    end
    
    return grad_σ
end

function calculate_discrete_gradient(fe::FerriteFESpace, a::AbstractVector, b::AbstractVector)
    grad_σ = zeros(fe.n)
    return calculate_discrete_gradient!(fe, grad_σ, a, b)
end

=#