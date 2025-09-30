mutable struct EITModeM
    u_g::AbstractVector
    u_g::AbstractVector
    b::AbstractVector
    λ::AbstractVector
    δσ::AbstractVector
    f::AbstractVector
    g::AbstractVector
    rhs::AbstractVector
    error::Float64
    length::Int64
    m::Int64
end
function EITModeN(g::AbstractVector, f::AbstractVector)
    L = length(g)
    M = length(f)
    return EITMode(zeros(L), zeros(M), zeros(L), zeros(L), f, g, zeros(L), 0.0, L, M)
end


function state_adjoint_step_mixed_cg!(mode::EITModeM, Ln::AbstractMatrix, Ld::AbstractMatrix, M, d, ∂d, down, up, fe::FerriteFESpace, maxiter=500)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u_g, Ln, mode.g; maxiter=maxiter)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : u = f
    cg!(mode.u_f, Ld, mode.f; maxiter=maxiter)

    mode.w = mode.u_f - u_g
    b = down(mode.w)
    mean = Statistics.mean(b)
    mode.w .-= mean

    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = ∂ₓd(w,0)
    cg!(mode.λ, L, ∂d(mode.w, 0); maxiter=maxiter)
    mode.error = d(mode.w, 0)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here:
    mode.δσ = calculate_bilinear_map!(mode.rhs, mode.λ, mode.w, fe, M)
    return mode.δσ, mode.error
end
