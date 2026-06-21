using LinearAlgebra
using Ferrite

export prox_tv_chambolle_pock

# 1. HELPER: Assemble the Weak Dual Force (No RT spaces needed!)
function assemble_weak_dual_force!(f_dual, p, cellvalues::CellValues, dh::DofHandler, ndims::Int64)
    fill!(f_dual, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    
    # We use a flat matrix layout for p: [ndims, n_qpoints, n_cells]
    for cell in CellIterator(dh)
        cell_idx = cellid(cell)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        
        fe = zeros(eltype(f_dual), n_basefuncs)
        
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            # Retrieve p for this specific cell and quad point
            pq = p[:, q, cell_idx] 
            
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                # Weak definition: ∫ p · ∇ϕ dΩ  (Note the sign change implicitly handles -div)
                fe[i] += dot(pq, ∇ϕᵢ) * dΩ
            end
        end
        f_dual[dofs] .+= fe
    end
    return f_dual
end

function prox_tv_chambolle_pock(y::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndims::Int64, M::AbstractArray; 
                                 λ=1.0, ρ=1.0, max_iter=200, tol=1e-5)
    α = λ / ρ
    
    # Step sizes (Crucial: σ * τ * L^2 < 1)
    τ = 0.1 
    σ = 0.1
    θ = 1.0
    
    # Build and factorize the LHS Primal Operator: (1 + τ)*M
    LHS = (1.0 + τ) * M
    LHS_fact = cholesky(LHS) # Fast back-solves every iteration
    
    # Initialize Primal Arrays
    z = copy(y)
    z_bar = copy(y)
    z_old = zeros(eltype(y), length(y))
    f_dual = zeros(eltype(y), length(y))
    rhs = zeros(eltype(y), length(y))
    
    # Initialize Dual Variable allocation flat layout
    num_cells = ndofs(dh) # Total cells check depending on structure, safer:
    num_cells = dh.grid.cells |> length 
    qpoints = getnquadpoints(cellvalues)
    
    # Allocation: [dimension, quad_points, cells]
    p = zeros(eltype(y), ndims, qpoints, num_cells)
    
    for iter in 1:max_iter
        copyto!(z_old, z)
        
        # 1. DUAL UPDATE & PROJECTION (Pointwise at Quad Points)
        for cell in CellIterator(dh)
            cell_idx = cellid(cell)
            reinit!(cellvalues, cell)
            dofs = celldofs(cell)
            ze = z_bar[dofs]
            
            for q in 1:qpoints
                # Evaluate continuous gradient at this specific quad point
                ∇z_q = zeros(eltype(y), ndims)
                for i in 1:getnbasefunctions(cellvalues)
                    ∇z_q .+= ze[i] * shape_gradient(cellvalues, q, i)
                end
                
                # Gradient ascent step
                p_proposed = p[:, q, cell_idx] + σ * ∇z_q
                
                # Project onto L2 ball (Isotropic TV)
                norm_p = norm(p_proposed)
                if norm_p > α
                    p[:, q, cell_idx] = p_proposed * (α / norm_p)
                else
                    p[:, q, cell_idx] = p_proposed
                end
            end
        end
        
        # 2. PRIMAL UPDATE 
        # Compute weak dual force: f_dual vector represents: M * (-div_weak(p))
        assemble_weak_dual_force!(f_dual, p, cellvalues, dh, ndims)
        
        # Construct RHS: M*z_old - τ*f_dual + τ*M*y
        # To avoid matrix allocations, handle via vectors:
        My = M * y
        Mz = M * z
        @. rhs = Mz - τ * f_dual + τ * My
        
        # Invert the mass system implicitly: z = LHS \ rhs
        ldiv!(z, LHS_fact, rhs)
        
        # 3. EXTRAPOLATION
        @. z_bar = z + θ * (z - z_old)
        
        # Convergence Check
        err = norm(z - z_old) / (norm(z) + 1e-10)
        if err < tol
            println("Converged in $iter iterations.")
            break
        end
    end
    
    return z
end

function prox_tv_chambolle_pock(y::AbstractVector, fe::FerriteFESpace; kwargs...)
    prox_tv_chambolle_pock(y, fe.cellvalues, fe.dh, fe.dim,fe.M; kwargs...)
end