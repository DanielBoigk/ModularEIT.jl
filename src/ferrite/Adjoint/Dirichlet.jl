function state_adjoint_step_neumann_cg!(mode::EITModeD, L::AbstractMatrix, M, d,∂d ,down,up,fe::FerriteFESpace, maxiter=500)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u,L, mode.g; maxiter = maxiter)
    b = down(mode.u)
    mean = Statistics.mean(b) 
    b .-= mean
    mode.u .-= mean 
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂u/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, L, up(∂d(b,mode.f)); maxiter = maxiter)
    mode.error = d(b,mode.f)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here: 
    mode.δσ = calculate_bilinear_map!(mode.rhs,mode.λ, mode.u, fe, M) 
    return mode.δσ, mode.error   
end