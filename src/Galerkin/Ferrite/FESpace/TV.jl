
export assemble_huber_gradient
export assemble_huber_hessian

"""
    normTV(a, cellvalues, dh, ndims, iso=2)

Compute the total variation (TV) seminorm of the field represented by coefficients `a`.

At each quadrature point, this computes
```math
‖∇u(x_q)‖_{iso}
````

and integrates over the domain.

# Arguments

* `a`: Coefficient vector of the FEM solution.
* `cellvalues`: Preallocated `CellValues` for quadrature and shape gradients.
* `dh`: `DofHandler` for iterating over elements.
* `ndims`: Number of spatial dimensions.
* `iso`: Norm degree.

  * `iso = 2` → isotropic TV
  * `iso = 1` → anisotropic TV

# Returns

Scalar value of the total variation seminorm.
"""
function normTV(a::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndims::Int64, iso=2)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    total_residual = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇uh_q = zeros(eltype(a), ndims)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                ∇uh_q .+= ue[i] * ∇ϕᵢ
            end
            total_residual += norm(∇uh_q, iso) * dΩ

        end
    end

    return total_residual
end


"""
    normTV_diff(a, cellvalues, dh, ndims; ε=1e-6, huber=huber_norm_smooth_isotropic)

Compute a *Huber-smoothed* total variation regularizer, differentiable w.r.t. `a`.

This replaces the nondifferentiable TV norm with a smooth approximation:
```math
‖∇u‖_ε ≈
  \begin{cases}
    \tfrac{1}{2ε}‖∇u‖^2, & ‖∇u‖ ≤ ε,\\
    ‖∇u‖ - \tfrac{ε}{2}, & ‖∇u‖ > ε.
  \end{cases}
````

# Arguments

* `a`: Coefficient vector of the FEM solution.
* `ε`: Huber smoothing threshold.
* `huber`: Function defining the smooth norm variant
  (`huber_norm_smooth_isotropic` or `huber_norm_smooth_anisotropic`).

# Returns

Scalar smoothed TV value suitable for differentiable regularization.

# Differentiation
This function is fully differentiable.
Gradients or Hessians with respect to `a` can be obtained using **Enzyme.jl**:
"""
function normTV_diff(a::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndims::Int64; ε::Float64=1e-6, huber=huber_norm_smooth_isotropic)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    total_residual = zero(eltype(a))
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇uh_q = zeros(eltype(a), ndims)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                ∇uh_q .+= ue[i] * ∇ϕᵢ
            end
            total_residual += huber(∇uh_q, ε) * dΩ
        end
    end
    return total_residual
end


"""
    normL1grad(fe, a)

L¹ norm of the gradient of the FE function represented by `a`.
Computes ∫ |∇u_h| dΩ over all elements using quadrature.
"""
function normTV(fe::FerriteFESpace, a::AbstractVector)
    normTV(a, fe.cellvalues, fe.dh, fe.dim)
end
function normTV_diff(fe::FerriteFESpace, a::AbstractVector)
    normTV_diff(a, fe.cellvalues, fe.dh, fe.dim)
end


function assemble_huber_gradient(a::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndim::Int, ϵ::Float64=1e-6)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    
    # Initialize the global gradient vector
    global_gradient = zeros(eltype(a), ndim)
    
    # Local element residual vector
    re = zeros(eltype(a), n_basefuncs)
    
    for cell in CellIterator(dh)
        fill!(re, 0.0)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            
            # 1. Compute ∇uh at quadrature point
            # Note: shape_gradient returns a Vec{ndims, T}
            ∇uh_q = function_gradient(cellvalues, q, ue)
            mag = norm(∇uh_q)
            
            # 2. Compute the Huber derivative vector (dv)
            if mag <= ϵ
                dv = ∇uh_q / ϵ
            else
                # Normalize ∇uh_q for the linear region derivative
                dv = ∇uh_q / mag 
            end
            
            # 3. Assemble local contribution
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                # The dot product of the Huber derivative and shape gradient
                re[i] += (dv ⋅ ∇ϕᵢ) * dΩ
            end
        end
        
        # Assemble into global vector
        global_gradient[dofs] += re
    end

    return global_gradient
end

function assemble_huber_gradient(fe::FerriteFESpace ,a::AbstractVector,ϵ::Float64=1e-6)
    assemble_huber_gradient(a, fe.cellvalues, fe.dh, fe.n, ϵ)
end
using Tensors, Ferrite
 

# Do not use except for very small problems !!!
function assemble_huber_hessian(a::AbstractVector, cellvalues::CellValues, dh::DofHandler, ndims::Int64, ϵ::Float64=1e-6)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    
    # Initialize global sparse matrix (Ferrite standard)
    K = allocate_matrix(dh) 
    # Local element stiffness matrix
    ke = zeros(eltype(a), n_basefuncs, n_basefuncs)
    # Identity tensor for ndims
    I_tensor = one(Tensor{2, ndims})

    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        ue = a[dofs]
        
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            ∇uh_q = function_gradient(cellvalues, q, ue)
            mag = norm(∇uh_q)
            
            # 1. Compute the Hessian of the Huber penalty (dH)
            if mag <= ϵ
                dH = I_tensor / ϵ
            else
                # Normalize gradient
                n_vec = ∇uh_q / mag
                # Projector: (I - n ⊗ n) / |∇u|
                dH = (I_tensor - n_vec ⊗ n_vec) / mag
            end
            
            # 2. Assemble local stiffness ke[i, j]
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                    # Triple product: ∇ϕᵢ' * dH * ∇ϕⱼ
                    ke[i, j] += (∇ϕᵢ ⋅ dH ⋅ ∇ϕⱼ) * dΩ
                end
            end
        end
        
        # Assemble local ke into global K
        assemble!(K, dofs, ke)
    end

    return K
end

function assemble_huber_hessian(fe::FerriteFESpace, a::AbstractVector, ϵ::Float64=1e-6)
    assemble_huber_hessian(a, fe.cellvalues, fe.dh, fe.n, ϵ)
end