using LinearAlgebra, Optimisers, LinearMaps, Ferrite

# Idea: have a struct that has everything preallocated and solves CG with multiple vectors at once.
# Solve the State equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
# Solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
# Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ)
#
# This is supposed to be very fast:
# Idea was to CG solve everything
mutable struct CGNeumannMatrix
    σ_outer::AbstractVector
    σ::AbstractVector

    K
    L₀::MatrixAMG
    L₁::MatrixAMG
    K_map #
    b_map::AbstractArray # preconditioned righhandside state equation
    λ_map::AbstractArray # preconditioned righthandside adjoint equation

    # Boundary data:
    F::Union{AbstractArray,Nothing} # This is the long vector for dirichlet boundary conditions
    f::Union{AbstractArray,Nothing} # This is the short vector for dirichlet boundary conditions
    G::Union{AbstractArray,Nothing} # This is the long vector for neumann boundary conditions
    g::Union{AbstractArray,Nothing} # This is the short vector for neumann boundary conditions

    # Helper arrays
    b::Union{AbstractArray,Nothing}

    # State equation:
    u::AbstractArray
    r::AbstractVector # the residual
    error::Number # sum of residuals

    # For CG solve
    u_x::Union{AbstractArray,Nothing}
    u_r::Union{AbstractArray,Nothing}
    u_p::Union{AbstractArray,Nothing}
    u_r²old::AbstractVector
    u_r²new::AbstractVector
    u_Ap::Union{AbstractArray,Nothing}
    u_α::AbstractVector

    error_upd::Bool

    # Adjoint equation:
    λ::AbstractArray
    λrhs::AbstractArray
    # For CG solve
    λ_x::Union{AbstractArray,Nothing}
    λ_r::Union{AbstractArray,Nothing}
    λ_p::Union{AbstractArray,Nothing}
    λ_r²old::Number
    λ_r²new::Number
    λ_Ap::Union{AbstractArray,Nothing}
    λ_α::Number


    # Functional derivative

    rhs::AbstractArray # This is a preallocation for calculating the bilinear map
    J::AbstractArray
    δσ::AbstractVector

    # Stuff for optimization:
    rule::OptRule # State of ADAM, ADAGrad, RMSProp or whatever
end
