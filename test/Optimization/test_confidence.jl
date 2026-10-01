# Confidence maps from the Jacobian: sensitivities, the diagonal of the model resolution matrix
# and the linearized posterior standard deviation.
using ModularEIT
using ModularEITFerrite
using Ferrite
using LinearAlgebra
using Random
using Test

struct LinearLSC <: AbstractObjective
    A::Matrix{Float64}
    b::Vector{Float64}
end
ModularEIT.objective_value(q::LinearLSC, x::AbstractVector) = sum(abs2, q.A * x - q.b) / 2
function ModularEIT.value_and_gradient!(g::AbstractVector, q::LinearLSC, x::AbstractVector)
    r = q.A * x - q.b
    g .= q.A' * r
    return sum(abs2, r) / 2
end
ModularEIT.n_residual(q::LinearLSC) = length(q.b)
ModularEIT.residual!(r::AbstractVector, q::LinearLSC, x::AbstractVector) = (r .= q.A * x .- q.b)
function ModularEIT.residual_and_jacobian!(r::AbstractVector, J::AbstractMatrix, q::LinearLSC, x::AbstractVector)
    r .= q.A * x .- q.b
    copyto!(J, q.A)
    return r, J
end

@testset "confidence maps" begin
    rng = MersenneTwister(43)

    @testset "linear problem: $name" for (name, m, n) in (("overdetermined", 30, 12), ("underdetermined", 12, 20))
        A = randn(rng, m, n) * Diagonal(10.0 .^ range(0, -3; length = n))
        q = LinearLSC(A, randn(rng, m))
        x = zeros(n)
        w = 0.5 .+ rand(rng, n)
        W = Diagonal(w)
        @test sensitivity_map(q, x) ≈ vec(sqrt.(sum(abs2, A; dims = 1)))
        @test sensitivity_map(q, x; weights = w) ≈ vec(sqrt.(sum(abs2, A; dims = 1))) ./ w

        js = jacobian_svd(q, x; weights = w)
        # truncation: R = V_k V_kᵀ W
        k = 5
        Rk = js.V[:, 1:k] * js.V[:, 1:k]' * W
        @test resolution_map(q, x; rank = k, weights = w) ≈ diag(Rk)
        @test resolution_map(js; rank = k) ≈ diag(Rk)               # reuse a computed SVD
        # Tikhonov filter: R = (AᵀA + λ s₁² W)⁻¹ AᵀA
        λ = 1e-3
        Rt = (A' * A + λ * js.s[1]^2 * W) \ (A' * A)
        @test resolution_map(js; λ) ≈ diag(Rt) atol = 1e-10
        for rm in (resolution_map(js; rank = k), resolution_map(js; λ), resolution_map(js; rtol = 1e-2))
            @test all(-1e-12 .<= rm .<= 1 + 1e-12)
        end
        m >= n && @test resolution_map(js; rank = n) ≈ ones(n)      # everything determined
        @test_throws ArgumentError resolution_map(js)
        @test_throws ArgumentError resolution_map(js; rank = 2, λ = 1.0)

        # linearized Gaussian posterior: prior N(x₀, γ² W⁻¹), noise N(0, η² I)
        γ, η = 0.7, 1e-2
        C = inv(A' * A / η^2 + W / γ^2)
        @test posterior_std(q, x; noise = η, prior_std = γ, weights = w) ≈ sqrt.(diag(C)) rtol = 1e-8
        @test posterior_std(js; noise = η, prior_std = γ) ≈ sqrt.(diag(C)) rtol = 1e-8
        @test all(posterior_std(js; noise = η, prior_std = γ) .<= γ ./ sqrt.(w) .+ 1e-12)
    end

    @testset "EIT: the boundary is better determined than the centre" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.1))
        I_ = trigonometric_patterns(fm, 5)
        V = forward_neumann(fm, ones(ndofs_σ(disc)), I_)[1]
        noise = RelativeGaussianNoise(0.01)
        data = AdjointStateObjective(fm, I_, add_noise(V, noise; rng))
        pp = PixelParametrization(disc, 16, 16)
        obj = ParametrizedObjective(data, pp)
        θ = ones(256)
        dist = [1 - max(abs(c[1]), abs(c[2])) for c in pp.centres]
        outer, inner = dist .< 0.25, dist .> 0.6
        mean(v) = sum(v) / length(v)
        js = jacobian_svd(obj, θ)
        for map in (sensitivity_map(obj, θ), resolution_map(js; rtol = 1e-2), resolution_map(js; λ = 1e-4))
            @test mean(map[outer]) > 3 * mean(map[inner])
        end
        # noise model → residual noise level; the posterior shrinks near the boundary only
        sd = posterior_std(obj, θ; noise, prior_std = 0.5)
        η = sqrt(2 * discrepancy_target(data, noise; τ = 1) / n_residual(data))
        @test sd ≈ posterior_std(js; noise = η, prior_std = 0.5)
        # (white pixel prior: single pixels are poorly determined, only combinations are, so the
        # pointwise shrinkage is moderate; the interior stays near the prior)
        @test mean(sd[outer]) < 0.9 * mean(sd[inner])
        @test mean(sd[inner]) > 0.9 * 0.5
        @test maximum(sd) <= 0.5 + 1e-12
        img = pixel_image(pp, resolution_map(js; rtol = 1e-2))
        @test size(img) == (16, 16)
    end
end
