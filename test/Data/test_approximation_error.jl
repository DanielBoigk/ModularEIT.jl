# Approximation-error model: modelling error of a coarse reconstruction model, estimated from
# samples, as Gaussian mean and low-rank covariance; whitened misfit, its derivatives and
# Gauss–Newton on it.
using ModularEIT
using Ferrite
using LinearAlgebra
using Random
using Statistics
using Test

@testset "approximation-error model" begin
    rng = MersenneTwister(71)

    @testset "whitening of noise plus low-rank modelling error" begin
        m, K, η = 40, 12, 0.3
        R = randn(rng, m, 3) * randn(rng, 3, K) .+ randn(rng, m)     # rank-3 variation around a mean
        ae = ApproximationError(R; noise = η, floor = 0)
        μ = vec(mean(R; dims = 2))
        @test ae.μ ≈ μ
        C = η^2 * I + cov(R; dims = 2)                                   # noise + sample covariance
        W = reduce(hcat, [whiten(ae, μ .+ e) for e in eachcol(Matrix(1.0I, m, m))])
        @test W * C * W' ≈ I atol = 1e-10
        @test W' * W ≈ inv(C) rtol = 1e-8
        @test whiten(ae, μ) ≈ zeros(m) atol = 1e-12
        @test_throws ArgumentError ApproximationError(R[:, 1:2]; noise = η)
        # the variance floor: a sample outside the span of the others is accounted for
        ν = 0.5
        R2 = randn(rng, m, 3) * randn(rng, 3, K) .+ ν .* randn(rng, m, K)
        ae2 = ApproximationError(R2; noise = η)
        @test ν < ae2.ν < 1.6ν                  # conservative: the others' span is itself noisy
        @test ae2.η ≈ sqrt(η^2 + ae2.ν^2)
    end

    # coarse reconstruction model, data from a fine mesh, little noise: the modelling error
    # dominates
    coarse = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
    fine = FerriteDiscretization(generate_grid(Quadrilateral, (48, 48)))
    els = angular_electrodes(coarse, 16; coverage = 0.5)
    fm = ForwardModel(coarse, CompleteElectrodeModel(els, 0.05))
    fmf = ForwardModel(fine, CompleteElectrodeModel(transfer_electrodes(coarse, els, fine), 0.05))
    I_ = trigonometric_patterns(fm, 5)
    pp = PixelParametrization(coarse, 8, 8)
    sample() = lognormal_phantom(gaussian_random_field(rng; ℓ = 0.4); s = 0.4)
    # residuals of the coarse model at the pixel values of samples, noise-free fine data
    function model_residual(ph)
        Vf = simulate_data(fine, fmf, ph, I_).clean
        obj = ParametrizedObjective(AdjointStateObjective(fm, I_, Vf), pp)
        return residual!(zeros(n_residual(obj)), obj, [ph(c) for c in pp.centres])
    end
    R = reduce(hcat, [model_residual(sample()) for _ in 1:60])
    truth = sample()
    noise = GaussianNoise(1e-5)
    V = simulate_data(fine, fmf, truth, I_; noise, rng).data
    data = ParametrizedObjective(AdjointStateObjective(fm, I_, V), pp)
    θtrue = [truth(c) for c in pp.centres]
    η = sqrt(2 * discrepancy_target(data.obj, noise; τ = 1) / n_residual(data))
    ae = ApproximationError(R; noise = η)
    obj = ApproximationErrorObjective(data, ae)

    @testset "the modelling error dominates, the model accounts for it" begin
        @test objective_value(data, θtrue) > 100 * discrepancy_target(data.obj, noise)
        # whitened residual of the truth: of the order of its expected size
        @test objective_value(obj, θtrue) < 5 * discrepancy_target(obj; τ = 1)
        @test discrepancy_target(obj; τ = 1) ≈ n_residual(obj) / 2
    end

    @testset "derivatives" begin
        θ = θtrue .* (1 .+ 0.1 .* randn(rng, length(θtrue)))
        m, n = n_residual(obj), length(θ)
        r, Jm = zeros(m), zeros(m, n)
        residual_and_jacobian!(r, Jm, obj, θ)
        @test r ≈ residual!(zeros(m), obj, θ)
        @test objective_value(obj, θ) ≈ sum(abs2, r) / 2
        δ = randn(rng, n)
        fd = (residual!(zeros(m), obj, θ .+ 1e-6 .* δ) - residual!(zeros(m), obj, θ .- 1e-6 .* δ)) / 2e-6
        @test Jm * δ ≈ fd rtol = 1e-5
        g = zeros(n)
        value_and_gradient!(g, obj, θ)
        @test g ≈ Jm' * r rtol = 1e-8
        J = jacobian_operator(obj, θ)
        w = randn(rng, m)
        @test J * δ ≈ Jm * δ rtol = 1e-8
        @test J' * w ≈ Jm' * w rtol = 1e-8
        @test jacobian_column_norms(obj, θ) ≈ vec(sqrt.(sum(abs2, Jm; dims = 1))) rtol = 1e-8
        G, gg = jacobian_gram(obj, θ)
        @test G ≈ Jm' * Jm rtol = 1e-8
        @test issymmetric(G)
        @test gg ≈ Jm' * r rtol = 1e-8
    end

    @testset "reconstruction: better than ignoring the modelling error" begin
        θ0 = fill(mean(θtrue), length(θtrue))
        err(θ) = norm(θ - θtrue) / norm(θtrue .- mean(θtrue))
        for solver in (:dense, :cg)
            gn = GaussNewton(; scaling = :sensitivity, linear_solver = solver)
            plain = minimize(data, θ0, gn; lower = 0.05, maxiter = 30, ftarget = discrepancy_target(data.obj, noise))
            aem = minimize(obj, θ0, gn; lower = 0.05, maxiter = 30, ftarget = discrepancy_target(obj))
            @test aem.status === :ftarget
            @test err(aem.σ) < err(plain.σ)
        end
    end
end
