# To be honest I don't know quite how this works. I don't know enough about CUDA, Reactant and the likes to properly implement it. Unless I have someone who cares and is willing to answer some questions I don't know how to properly do that.

# The goal is to have a Conjugate Gradient like solver loop that simultaneously updates state solution and adjoint solution and gradient. ADAM or similar optimizers manage the steps size and send gradients back to CPU.
# The CPU manages updates to σ and assembles a new matrix from the snapshot.
# It should in theory be super fast. But the problem is that the adjoint state equation currently requires a Ferrite assembly loop to aquire: ∇(u) ⋅ ∇(λ) which doesn't play well with the GPU.

using LinearAlgebra, Optimisers

mutable struct CGNeumannEITMode{OptRule} where {OptRule<:Optimisers.AbstractRule}
    # Boundary data:
    F::Union{AbstractVector,Nothing} # This is the long vector for dirichlet boundary conditions
    f::Union{AbstractVector,Nothing} # This is the short vector for dirichlet boundary conditions
    G::Union{AbstractVector,Nothing} # This is the long vector for neumann boundary conditions
    g::Union{AbstractVector,Nothing} # This is the short vector for neumann boundary conditions

    # Helper arrays
    b::Union{AbstractVector,Nothing}

    # State equation:
    u::AbstractVector
    error::Number
    # For CG solve
    u_x::Union{AbstractVector,Nothing}
    u_r::Union{AbstractVector,Nothing}
    u_p::Union{AbstractVector,Nothing}
    u_r²old::Number
    u_r²new::Number
    u_Ap::Union{AbstractVector,Nothing}
    u_α::Number

    error_upd::Bool

    # Adjoint equation:
    λ::AbstractVector
    λrhs::AbstractVector
    # For CG solve
    λ_x::Union{AbstractVector,Nothing}
    λ_r::Union{AbstractVector,Nothing}
    λ_p::Union{AbstractVector,Nothing}
    λ_r²old::Number
    λ_r²new::Number
    λ_Ap::Union{AbstractVector,Nothing}
    λ_α::Number


    # Functional derivative

    rhs::AbstractVector # This is a preallocation for calculating the bilinear map
    δσ::AbstractVector

    # Stuff for optimization:
    rule::OptRule # State of ADAM, ADAGrad, RMSProp or whatever
end

# do not export:
function conjugate_gradient_reference(
    A, b::AbstractVector, x0::AbstractVector=zero(b); atol=length(b) * eps(norm(b))
)
    x = copy(x0)                        # initialize the solution
    r = b - A * x0                      # initial residual
    p = copy(r)                         # initial search direction
    r²old = r' * r                      # squared norm of residual

    k = 0
    while r²old > atol^2                # iterate until convergence
        Ap = A * p                      # search direction
        α = r²old / (p' * Ap)           # step size
        @. x += α * p                   # update solution
        # Update residual:
        if (k + 1) % 16 == 0            # every 16 iterations, recompute residual from scratch
            r .= b .- A * x             # to avoid accumulation of numerical errors
        else
            @. r -= α * Ap              # use the updating formula that saves one matrix-vector product
        end
        r²new = r' * r
        @. p = r + (r²new / r²old) * p  # update search direction
        r²old = r²new                   # update squared residual norm
        k += 1
    end

    return x
end

function CGNeumannEITMode(fe::FerriteFESpace, f_vec::AbstractVector, g_vec::AbstractVector, rule)
    if length(f_vec) == fe.n
        F = copy(f_vec)
        f = fe.down(F)
        mean_f = Statistics.mean(f)
        f .-= mean_f
        F .-= mean_f
    elseif length(f_vec) == fe.m
        f = copy(f_vec)
        mean_f = Statistics.mean(f)
        f .-= mean_f
        F = fe.up(f)
    end
    if length(g_vec) == fe.n
        G = copy(g_vec)
        g = fe.down(G)
        mean_g = Statistics.mean(g)
        g .-= mean_g
        G .-= mean_g
    elseif length(g_vec) == fe.m
        g = copy(g_vec)
        mean_g = Statistics.mean(g)
        g .-= mean_g
        G = fe.up(g)
    end



end

function CG_init(mode::CGNeumannEITMode, sol::FerriteSolverState, fe::FerriteFESpace)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L_fac
    down = fe.down
    up = fe.up

    mode.u .= L \ mode.G
    # Normalize
    mean_boundary!(mode.u_g, mode, down)
    mode.error = d(mode.b, mode.f)
    mode.λrhs .= up(∂d(mode.b, mode.f))
    mean_boundary!(mode.λrhs, mode, down)
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂λ/∂𝐧 = ∂ₓd(u,f)
    mode.λ = L \ mode.λrhs
    mode.λ .-= Statistics.mean(mode.λ)
    # Calculate ∂J(σ,f,g)/∂σ = ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = -calculate_bilinear_map!(fe, mode.rhs, mode.λ, mode.u)
end

function objective_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u_g, L, mode.G; maxiter=maxiter)
    # Normalize
    mean_boundary!(mode.u_g, mode, down)
    mode.error_n = d(mode.b, mode.f)
    return mode.error_n
end

function ask_objective()


end
#
function CG_state_step!(L::AbstractArray, mode::CGNeumannEITMode, fe::FerriteFESpace; update_r::Bool=false, update_err::Bool=false, mean_func! = mean_boundary!)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up

    mode.u_Ap = L * mode.u_p                      # search direction
    mode.u_α = mode.u_r²old / (mode.u_p' * mode.u_Ap)           # step size
    @. mode.u += mode.u_α * mode.u_p                   # update solution
    # Update residual:
    if update_r
        mode.u_r .= mode.G .- L * mode.u             # to avoid accumulation of numerical errors
    else
        @. mode.u_r -= mode.u_α * mode.u_Ap              # use the updating formula that saves one matrix-vector product
    end
    mode.u_r²new = mode.u_r' * mode.u_r
    @. mode.u_p = mode.u_r + (mode.u_r²new / mode.u_r²old) * mode.u_p  # update search direction
    mode.u_r²old = mode.u_r²new                   # update squared residual norm
    mode.u_k += 1
    mode.error_upd = false
end

function objective_neumann_cg!(mode::FerriteEITMode, sol::FerriteSolverState, fe::FerriteFESpace, maxiter=500)
    d = sol.d
    ∂d = sol.∂d
    L = sol.L
    down = fe.down
    up = fe.up
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u_g, L, mode.G; maxiter=maxiter)
    # Normalize
    mean_boundary!(mode.u_g, mode, down)
    mode.error_n = d(mode.b, mode.f)
    return mode.error_n
end

function CG_adjoint_step!(K::AbstractArray, mode::CGNeumannEITMode, fe::FerriteFESpace)
    mode.λ_Ap = L * mode.λ_p                      # search direction
    mode.λ_α = mode.λ_r²old / (mode.λ_p' * mode.λ_Ap)           # step size
    @. mode.λ += mode.λ_α * mode.λ_p                   # update solution
    # Update residual:
    if update
        mode.λ_r .= mode.λrhs .- L * mode.λ             # to avoid accumulation of numerical errors
    else
        @. mode.λ_r -= mode.λ_α * mode.λ_Ap              # use the updating formula that saves one matrix-vector product
    end
    mode.λ_r²new = mode.λ_r' * mode.λ_r
    @. mode.λ_p = mode.λ_r + (mode.λ_r²new / mode.λ_r²old) * mode.λ_p  # update search direction
    mode.λ_r²old = mode.λ_r²new                   # update squared residual norm
    mode.λ_k += 1
end

function CG_functional_derivative!(K::AbstractArray, mode::CGNeumannEITMode, fe::FerriteFESpace)

end
