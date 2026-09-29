# Proximal operators (Tikhonov, smoothed and exact total variation, user maps) and the proximal
# methods built on them: proximal gradient (FISTA) and ADMM.
using ModularEIT
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

# ½ (x - c)ᵀ A (x - c) (same as in test_optimizers.jl, without residuals)
struct ProxQuadratic <: AbstractObjective
    A::Matrix{Float64}
    c::Vector{Float64}
end
ModularEIT.objective_value(q::ProxQuadratic, x::AbstractVector) = dot(x - q.c, q.A, x - q.c) / 2
function ModularEIT.value_and_gradient!(g::AbstractVector, q::ProxQuadratic, x::AbstractVector)
    g .= q.A * (x - q.c)
    return dot(x - q.c, g) / 2
end

# ½‖x - c‖² in residual form r = x - c (for Gauss–Newton)
struct QuadraticResidual <: AbstractObjective
    c::Vector{Float64}
end
ModularEIT.objective_value(q::QuadraticResidual, x::AbstractVector) = sum(abs2, x - q.c) / 2
function ModularEIT.value_and_gradient!(g::AbstractVector, q::QuadraticResidual, x::AbstractVector)
    g .= x .- q.c
    return sum(abs2, g) / 2
end
ModularEIT.n_residual(q::QuadraticResidual) = length(q.c)
ModularEIT.residual!(r::AbstractVector, q::QuadraticResidual, x::AbstractVector) = (r .= x .- q.c)
function ModularEIT.residual_and_jacobian!(r::AbstractVector, J::AbstractMatrix, q::QuadraticResidual, x::AbstractVector)
    r .= x .- q.c
    copyto!(J, I)
    return r, J
end

# value of the prox problem R(z) + ρ/2 Σ wᵢ (zᵢ - vᵢ)²
prox_value(reg, z, v, ρ, w) = objective_value(reg, z) + ρ / 2 * sum(w .* (z .- v) .^ 2)

# convex problem: no feasible random perturbation may improve the value
function no_better_neighbour(reg, z, v, ρ, w, lo, hi; rng, trials = 30, scale = 1e-3)
    P = prox_value(reg, z, v, ρ, w)
    all(1:trials) do _
        zt = clamp.(z .+ scale .* randn(rng, length(z)), lo, hi)
        prox_value(reg, zt, v, ρ, w) >= P - 1e-10 * max(1, abs(P))
    end
end

@testset "proximal operators and methods" begin
    rng = MersenneTwister(21)
    q0 = FerriteDiscretization(generate_grid(Quadrilateral, (10, 10)))
    p1 = FerriteDiscretization(generate_grid(Triangle, (6, 6)); ip_σ = Lagrange{RefTriangle, 1}())

    @testset "lumped mass" begin
        mats = FEMatrices(p1)
        m = lumped_mass(p1)
        @test m ≈ vec(sum(mats.M_σ; dims = 2))
        @test sum(m) ≈ 4                                 # area of [-1, 1]²
        @test lumped_mass(q0) ≈ diag(FEMatrices(q0).M_σ)
    end

    @testset "Tikhonov prox ($(isnothing(w) ? "Euclidean" : "weighted"))" for w in (nothing, :mass)
        disc = q0
        n = ndofs_σ(disc)
        wv = w === nothing ? ones(n) : lumped_mass(disc)
        reg = TikhonovRegularizer(disc; kind = :jump, reference = 0.5)
        v = randn(rng, n)
        ρ = 3.0
        z = prox(reg, v, ρ; weights = w === nothing ? nothing : wv)
        g = zeros(n)
        value_and_gradient!(g, reg, z)
        @test norm(g .+ ρ .* wv .* (z .- v)) < 1e-10      # optimality
        # with bounds: convex, so no feasible neighbour is better
        zb = prox(reg, v, ρ; weights = wv, lower = -0.2, upper = 0.3)
        @test all(-0.2 .<= zb .<= 0.3)
        @test no_better_neighbour(reg, zb, v, ρ, wv, -0.2, 0.3; rng)
    end

    @testset "smoothed TV prox" begin
        n = ndofs_σ(q0)
        reg = TotalVariationRegularizer(q0; ε = 1e-2)
        v = randn(rng, n)
        z = prox(reg, v, 5.0)
        g = zeros(n)
        value_and_gradient!(g, reg, z)
        @test norm(g .+ 5.0 .* (z .- v)) < 1e-8
    end

    @testset "exact TV prox of a step (P0, $(isnothing(w) ? "Euclidean" : "weighted"))" for w in (nothing, :mass)
        # data a | b with a vertical interface: the ROF solution keeps the two plateaus and moves
        # them towards each other by α L / (ρ A) (L interface length, A side "areas" in the metric)
        disc = q0
        n = ndofs_σ(disc)
        wv = w === nothing ? ones(n) : lumped_mass(disc)
        v = interpolate_function(disc, x -> x[1] < 0 ? 1.0 : 2.0)
        left = v .== 1.0
        reg = TotalVariationRegularizer(disc; ε = 0)
        ρ = 20.0
        z = prox(reg, v, ρ; weights = w === nothing ? nothing : wv)
        L = 2.0
        Al, Ar = sum(wv[left]), sum(wv[.!left])
        @test z[left] ≈ fill(1 + L / (ρ * Al), count(left)) atol = 1e-6
        @test z[.!left] ≈ fill(2 - L / (ρ * Ar), count(.!left)) atol = 1e-6
        # strong regularization merges the plateaus into the weighted mean
        zm = prox(reg, v, 1e-3; weights = w === nothing ? nothing : wv)
        @test zm ≈ fill(dot(wv, v) / sum(wv), n) atol = 1e-6
    end

    @testset "exact TV prox with bounds and continuous σ" begin
        for disc in (q0, p1)
            n = ndofs_σ(disc)
            wv = lumped_mass(disc)
            reg = TotalVariationRegularizer(disc; ε = 0)
            v = randn(rng, n)
            z = prox(reg, v, 10.0; weights = wv, lower = -0.5, upper = 0.8)
            @test all(-0.5 .<= z .<= 0.8)
            @test no_better_neighbour(reg, z, v, 10.0, wv, -0.5, 0.8; rng)
        end
    end

    @testset "user-defined proximal map" begin
        # soft thresholding: prox of ‖z‖₁ in the Euclidean metric
        pm = ProximalMap((z, v, ρ) -> (z .= sign.(v) .* max.(abs.(v) .- 1 / ρ, 0)))
        @test pm isa AbstractRegularizer
        @test prox(pm, [3.0, -0.2, -2.0], 2.0) ≈ [2.5, 0.0, -1.5]
        @test prox(pm, [3.0, -0.2, -2.0], 2.0; upper = 1.0) ≈ [1.0, 0.0, -1.5]
    end

    # min ½‖x - c‖² + α TV(x): the solution is the TV prox of c with ρ = 1/α (Euclidean)
    @testset "proximal methods reproduce the prox: $name" for (name, mk) in
            (("proximal gradient", reg -> ProximalGradient(0.1 => reg)),
             ("proximal gradient (no acceleration)", reg -> ProximalGradient(0.1 => reg; accelerated = false)),
             ("ADMM", reg -> ADMM(0.1 => reg; ρ = 1.0)),
             ("ADMM + Gauss–Newton inner", reg -> ADMM(0.1 => reg; ρ = 1.0, inner = GaussNewton())))
        disc = q0
        n = ndofs_σ(disc)
        c = interpolate_function(disc, x -> x[1] < 0 ? 1.0 : 2.0) .+ 0.1 .* randn(rng, n)
        reg = TotalVariationRegularizer(disc; ε = 0)
        exact = prox(reg, c, 10.0)
        f = name == "ADMM + Gauss–Newton inner" ? QuadraticResidual(c) : ProxQuadratic(Matrix(1.0I, n, n), c)
        # gtol above the accuracy of the inner (iterative) TV prox, ~1e-8 relative here
        res = minimize(f, zeros(n), mk(reg); maxiter = 3000, gtol = 1e-7)
        @test res.converged
        @test res.σ ≈ exact atol = 1e-4
        # with bounds
        exact_b = prox(reg, c, 10.0; lower = 1.2, upper = 1.9)
        res = minimize(f, fill(1.5, n), mk(reg); lower = 1.2, upper = 1.9, maxiter = 3000, gtol = 1e-7)
        @test all(1.2 .<= res.σ .<= 1.9)
        @test res.σ ≈ exact_b atol = 1e-4
    end

    @testset "EIT with total variation: $name" for (name, method) in
            (("proximal gradient", :pg), ("ADMM", :admm))
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        mats = FEMatrices(disc)
        n = ndofs_σ(disc)
        fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.1))
        σtrue = interpolate_function(disc, x -> 1 + (norm(x - Vec(0.3, 0.2)) < 0.4))
        I_ = trigonometric_patterns(fm, 7)
        V = forward_neumann(fm, σtrue, I_)[1]
        data = AdjointStateObjective(fm, I_, V)
        tv = TotalVariationRegularizer(disc; ε = 0)
        w = lumped_mass(disc)
        opt = method === :pg ? ProximalGradient(1e-5 => tv; weights = w) :
              ADMM(1e-5 => tv; ρ = 1e-2, weights = w, inner = GaussNewton(), inner_maxiter = 3)
        σ0 = ones(n)
        J0 = objective_value(data, σ0)
        res = minimize(data, σ0, opt; lower = 0.1, maxiter = method === :pg ? 300 : 40)
        @test all(res.σ .>= 0.1)
        @test objective_value(data, res.σ) < 1e-2 * J0
        @test fe_norm(disc, res.σ - σtrue; mats) < 0.8 * fe_norm(disc, σ0 - σtrue; mats)
    end
end
