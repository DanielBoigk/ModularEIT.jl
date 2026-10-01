# Noise models: statistics, reproducibility, source/meter noise through the forward model,
# operator-level noise, modelling errors and the discrepancy target.
using ModularEIT
using ModularEITFerrite
using Ferrite
using LinearAlgebra
using Statistics
using Random
using Test

@testset "noise models" begin
    @testset "absolute Gaussian noise" begin
        clean = zeros(200, 300)
        noisy = add_noise(clean, GaussianNoise(0.1); rng = MersenneTwister(1))
        @test size(noisy) == size(clean)
        @test clean == zeros(200, 300)                         # not modified
        @test std(noisy) ≈ 0.1 rtol = 0.02
        @test abs(mean(noisy)) < 2e-3
        # reproducible
        @test add_noise(clean, GaussianNoise(0.1); rng = MersenneTwister(1)) == noisy
        # one standard deviation per measurement channel (row)
        s = collect(range(0.01, 0.2; length = 200))
        noisy = add_noise(clean, GaussianNoise(s); rng = MersenneTwister(2))
        @test vec(std(noisy; dims = 2)) ≈ s rtol = 0.2
        @test_throws DimensionMismatch add_noise(zeros(3, 2), GaussianNoise([1.0, 2.0]))
        @test_throws ArgumentError GaussianNoise(-1.0)
        # in place
        x = zeros(10)
        add_noise!(x, GaussianNoise(1.0); rng = MersenneTwister(3))
        @test x == add_noise(zeros(10), GaussianNoise(1.0); rng = MersenneTwister(3))
    end

    @testset "relative Gaussian noise" begin
        rng = MersenneTwister(4)
        clean = randn(rng, 400, 50) .* (1:50)'                # columns of different size
        δ = 0.05
        noisy = add_noise(clean, RelativeGaussianNoise(δ); rng)
        ratios = [norm(noisy[:, k] - clean[:, k]) / norm(clean[:, k]) for k in 1:50]
        @test mean(ratios) ≈ δ rtol = 0.02                      # ‖ηᵢ‖ ≈ δ ‖fᵢ‖
        @test maximum(abs.(ratios .- δ)) < 0.2δ
        @test expected_squared_error(RelativeGaussianNoise(δ), clean) ≈ δ^2 * sum(abs2, clean)
        @test expected_squared_error(GaussianNoise(0.1), clean) ≈ 0.01 * length(clean)
    end

    # CEM on a mesh, used by the tests below
    disc = FerriteDiscretization(generate_grid(Quadrilateral, (12, 12)))
    fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.1))
    σ = ones(ndofs_σ(disc))
    I_ = trigonometric_patterns(fm, 3)

    @testset "source and meter noise" begin
        noise = SourceMeterNoise(0.05, 0.0)
        res = simulate_data(disc, fm, σ, I_; noise, rng = MersenneTwister(5))
        @test res.inputs == I_                                  # nominal inputs are returned
        @test res.applied != I_
        w = vec(sum(fm.P; dims = 1))
        @test norm(w' * res.applied) < 1e-12                    # applied currents: zero net current
        @test res.data ≈ forward_neumann(fm, σ, res.applied)[1]  # no meter noise
        @test res.clean ≈ forward_neumann(fm, σ, I_)[1]
        # meter noise only
        res = simulate_data(disc, fm, σ, I_; noise = SourceMeterNoise(0.0, 0.01), rng = MersenneTwister(6))
        @test res.applied == I_
        @test 0.005 < std(res.data - res.clean) < 0.02
        # voltage-driven: source noise on the voltages, meter noise on the currents
        V_ = trigonometric_patterns(fm, 2)
        res = simulate_data(disc, fm, σ, V_; mode = :dirichlet, noise = SourceMeterNoise(0.02, 0.0),
                            rng = MersenneTwister(7))
        @test res.data ≈ forward_dirichlet(fm, σ, res.applied)[1]
        @test_throws ArgumentError add_noise(res.clean, SourceMeterNoise(0.1, 0.1))
    end

    @testset "operator-level noise" begin
        R = Symmetric(randn(MersenneTwister(8), 6, 6)) |> Matrix
        Rn = perturb_boundary_operator(R, 0.3; rng = MersenneTwister(9))
        @test issymmetric(Rn)
        # E = (Ê + Êᵀ)/2: off-diagonal variance s²/2, diagonal variance s²
        samples = [perturb_boundary_operator(zeros(40, 40), 1.0; rng = MersenneTwister(k)) for k in 1:20]
        off = [S[i, j] for S in samples for i in 1:40 for j in 1:40 if i != j]
        dia = [S[i, i] for S in samples for i in 1:40]
        @test var(off) ≈ 0.5 rtol = 0.05
        @test var(dia) ≈ 1.0 rtol = 0.15
    end

    @testset "modelling errors" begin
        els = angular_electrodes(disc, 8; coverage = 0.5)
        cem = CompleteElectrodeModel(els, 0.1)
        pert = perturb_contact_impedance(cem, 0.2; rng = MersenneTwister(10))
        @test pert.electrodes == cem.electrodes
        @test all(pert.z .> 0)
        @test pert.z != cem.z
        @test std(log.(pert.z ./ cem.z)) < 0.5
        # electrode positions: explicit (jittered) centre angles
        θ = 2π .* (0:7) ./ 8
        @test angular_electrodes(disc, 8; coverage = 0.5, angles = θ) == els
        θj = electrode_angles(8; jitter = 0.15, rng = MersenneTwister(11))
        @test length(θj) == 8
        @test maximum(abs.(θj .- θ)) < 5 * 0.15
        @test angular_electrodes(disc, 8; coverage = 0.5, angles = θj) != els
    end

    @testset "discrepancy target matches the objective's misfit" begin
        # E[J(σ_true)] for noisy data equals the target / τ²
        noise = GaussianNoise(1e-3)
        clean = forward_neumann(fm, σ, I_)[1]
        Js = map(1:300) do k
            data = add_noise(clean, noise; rng = MersenneTwister(100 + k))
            objective_value(AdjointStateObjective(fm, I_, data), σ)
        end
        obj = AdjointStateObjective(fm, I_, add_noise(clean, noise; rng = MersenneTwister(1)))
        @test discrepancy_target(obj, noise; τ = 1.0) ≈ mean(Js) rtol = 0.1
        @test discrepancy_target(obj, noise; τ = 1.2) ≈ 1.44 * discrepancy_target(obj, noise; τ = 1.0)
        # weighted misfit: the target is taken in the misfit's metric
        W = Diagonal(fill(4.0, n_measure(fm)))
        objw = AdjointStateObjective(fm, I_, obj.data; misfit = WeightedSquaredEuclidean(W))
        @test discrepancy_target(objw, noise; τ = 1.0) ≈ 4 * discrepancy_target(obj, noise; τ = 1.0)
    end
end
