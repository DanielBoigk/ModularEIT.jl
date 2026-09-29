# Boundary-data SVD truncated at the noise level: noise of the rotated data, noise level of every
# new pattern, truncation and reconstruction from the retained pairs.
using ModularEIT
using Ferrite
using LinearAlgebra
using Random
using Test

@testset "pattern SVD truncated at the noise level" begin
    rng = MersenneTwister(53)

    @testset "per-entry Gaussian noise" begin
        s = rand(rng, 4, 3)
        noisy = add_noise(zeros(4, 3), GaussianNoise(s); rng)
        @test size(noisy) == (4, 3)
        @test_throws DimensionMismatch add_noise(zeros(3, 3), GaussianNoise(s))
        # expected misfit: sum of the variances
        @test ModularEIT.expected_squared_error(GaussianNoise(s), zeros(4, 3)) ≈ sum(abs2, s)
    end

    disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
    fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.1))
    Gt = trigonometric_patterns(fm, 7)
    s = size(Gt, 2)
    G = Gt * (I + 0.3 * randn(rng, s, s))                # far from orthonormal
    σtrue = conductivity(disc, InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.4, 2.5)]))
    clean = forward_neumann(fm, σtrue, G)[1]
    cleant = forward_neumann(fm, σtrue, Gt)[1]
    σref = ones(ndofs_σ(disc))

    @testset "noise of the rotated data ($metric)" for metric in (:euclidean, :L2)
        for noise in (GaussianNoise(1e-3), RelativeGaussianNoise(0.01))
            V = add_noise(clean, noise; rng)
            p = pattern_svd(disc, fm, G, V; metric, noise)
            C = p.combination
            @test p.currents ≈ G * C
            @test p.voltages ≈ reground(V, p.Mv * ones(size(V, 1))) * C
            # per-entry variances of E C for independent entries of E
            S = abs2.(ModularEIT._noise_std(noise, V)) .* ones(size(V))
            @test p.noise isa GaussianNoise
            @test abs2.(p.noise.std) ≈ S * abs2.(C)
            # noise level of pattern k: sqrt(E ‖reground(E) c_k‖²_Mv), Monte Carlo check
            Lv = cholesky(Symmetric(Matrix(p.Mv))).L
            n_mc = 4000
            acc = zeros(s)
            for _ in 1:n_mc
                E = sqrt.(S) .* randn(rng, size(V))
                acc .+= vec(sum(abs2, Lv' * reground(E, p.Mv * ones(size(V, 1))) * C; dims = 1))
            end
            @test p.noise_levels ≈ sqrt.(acc ./ n_mc) rtol = 0.05
        end
    end

    @testset "difference to a reference" begin
        V = add_noise(cleant, RelativeGaussianNoise(0.01); rng)
        Vref = forward_neumann(fm, σref, Gt)[1]
        p = pattern_svd(disc, fm, Gt, V; metric = :L2, reference = σref)
        @test p.reference ≈ pattern_svd(disc, fm, Gt, V; metric = :L2, reference = Vref).reference
        w = p.Mv * ones(size(V, 1))
        @test p.voltages ≈ reground(V, w) * p.combination     # measured data, rotated
        @test p.reference ≈ reground(Vref, w) * p.combination
        D = p.voltages - p.reference
        @test D' * p.Mv * D ≈ Diagonal(p.values .^ 2) atol = 1e-10 * p.values[1]^2
        @test issorted(p.values; rev = true)
        # absolute data: relative noise gives signal/noise ≈ 1/δ in every pattern, nothing to cut
        pa = pattern_svd(disc, fm, Gt, V; metric = :L2, noise = RelativeGaussianNoise(0.01))
        @test all(pa.values ./ pa.noise_levels .> 30)
        @test size(truncate_patterns(pa; τ = 2).currents, 2) == s
    end

    @testset "truncation" begin
        δ = 0.01
        V = add_noise(cleant, RelativeGaussianNoise(δ); rng)
        p = pattern_svd(disc, fm, Gt, V; metric = :L2, noise = RelativeGaussianNoise(δ), reference = σref)
        K = findfirst(p.values .<= 2 .* p.noise_levels)
        K = K === nothing ? s : K - 1
        t = truncate_patterns(p)
        @test size(t.currents, 2) == size(t.voltages, 2) == length(t.values) == K
        @test t.currents == p.currents[:, 1:K] && t.voltages == p.voltages[:, 1:K]
        @test t.noise.std == p.noise.std[:, 1:K] && t.noise_levels == p.noise_levels[1:K]
        @test t.reference == p.reference[:, 1:K]
        # only the leading patterns distinguish the inclusion from the reference above the noise
        @test 2 <= K <= s ÷ 2
        t3 = truncate_patterns(p, 3)
        @test size(t3.currents, 2) == 3
        @test_throws ArgumentError truncate_patterns(p, s + 1)
        # more noise → fewer pairs
        Vn = add_noise(cleant, RelativeGaussianNoise(8δ); rng)
        pn = pattern_svd(disc, fm, Gt, Vn; metric = :L2, noise = RelativeGaussianNoise(8δ), reference = σref)
        @test size(truncate_patterns(pn).currents, 2) < K
        # without a noise model there are no noise levels to truncate at
        @test_throws ArgumentError truncate_patterns(pattern_svd(disc, fm, Gt, V); τ = 2)

        # reconstruction from the retained pairs, stopped at their noise level
        obj = AdjointStateObjective(fm, t.currents, t.voltages)
        target = discrepancy_target(obj, t.noise)
        @test target > 0
        σ0 = ones(ndofs_σ(disc))
        res = minimize(obj, σ0, TruncatedGaussNewton(; rtol = 1e-2); lower = 0.1, maxiter = 30, ftarget = target)
        @test res.status === :ftarget
        @test norm(res.σ - σtrue) < norm(σ0 - σtrue)
    end
end
