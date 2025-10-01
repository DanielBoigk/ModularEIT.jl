

# Not yet correct
# Replace with SmoothOptimizers.jl
function split_bregman(γ0, y; λ=1.0, μ=1.0, τ=0.1, niter=20, prox)
    γ = copy(γ0)
    d = zeros(size(γ0))
    b = zeros(size(γ0))

    for k in 1:niter
        # --- Update γ (data term + quadratic penalty) --
        # Solve: min 0.5||F(γ)-y||^2 + μ/2 ||∇γ - d - b||^2
        g = grad_data(γ, y) + μ * (∇(γ) - d - b)
        γ -= τ * g # step size

        # --- Update auxiliary variable d (prox of regularizer) ---
        d = prox(γ + b, λ/μ)

        # --- Update Bregman variable ---
        b += γ - d
    end

    return γ
end
