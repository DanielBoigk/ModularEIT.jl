# Matrix-free Jacobians (J v, Jᵀ w), Jacobians in row blocks (column norms without storing J),
# and Gauss–Newton with CG on the normal equations.
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

@testset "matrix-free Gauss–Newton" begin
    rng = MersenneTwister(61)
    disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
    fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.1))
    I_ = trigonometric_patterns(fm, 5)
    σtrue = conductivity(disc, InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.4, 2.5)]))
    noise = RelativeGaussianNoise(0.005)
    V = add_noise(forward_neumann(fm, σtrue, I_)[1], noise; rng)
    pp = PixelParametrization(disc, 16, 16)
    n_obs = size(V, 1)
    U = Matrix(qr(randn(rng, n_obs, n_obs)).Q)[1:(n_obs - 4), :]
    objectives = [
        ("σ, Euclidean", AdjointStateObjective(fm, I_, V), ones(ndofs_σ(disc)) .+ 0.2 .* rand(rng, ndofs_σ(disc))),
        ("σ, projected misfit", AdjointStateObjective(fm, I_, V; misfit = ProjectedMisfit(U)), 1 .+ 0.2 .* rand(rng, ndofs_σ(disc))),
        ("pixels", ParametrizedObjective(AdjointStateObjective(fm, I_, V), pp), 1 .+ 0.2 .* rand(rng, 256)),
    ]

    @testset "J v and Jᵀ w: $name" for (name, obj, θ) in objectives
        m, n = n_residual(obj), length(θ)
        r, Jm = zeros(m), zeros(m, n)
        residual_and_jacobian!(r, Jm, obj, θ)
        J = jacobian_operator(obj, θ)
        @test size(J) == (m, n)
        v, w = randn(rng, n), randn(rng, m)
        @test J * v ≈ Jm * v rtol = 1e-8
        @test J' * w ≈ Jm' * w rtol = 1e-8
        y = zeros(m)
        mul!(y, J, v)
        @test y ≈ Jm * v rtol = 1e-8
        # column norms (sensitivities) from row blocks, without storing J
        @test jacobian_column_norms(obj, θ) ≈ vec(sqrt.(sum(abs2, Jm; dims = 1))) rtol = 1e-10
        # the Gram matrix JᵀJ and Jᵀr from row blocks
        G, g = jacobian_gram(obj, θ)
        @test G ≈ Jm' * Jm rtol = 1e-10
        @test issymmetric(G)                               # (accumulated in one triangle)
        @test g ≈ Jm' * r rtol = 1e-10
        @test sensitivity_map(obj, θ) ≈ vec(sqrt.(sum(abs2, Jm; dims = 1))) rtol = 1e-10
    end

    @testset "Gauss–Newton with CG: $name" for (name, reg) in (("no penalty", nothing),
                                                              ("TV", 1e-5 => TotalVariationRegularizer(pp.pixel_disc; ε = 1e-2)))
        data = ParametrizedObjective(AdjointStateObjective(fm, I_, V), pp)
        obj = reg === nothing ? data : RegularizedObjective(data, reg)
        target = discrepancy_target(data.obj, noise)
        θ0 = ones(256)
        for scaling in (:identity, :sensitivity)
            dense = minimize(obj, θ0, GaussNewton(; scaling, linear_solver = :dense); lower = 0.1, maxiter = 15)
            cg = minimize(obj, θ0, GaussNewton(; scaling, linear_solver = :cg); lower = 0.1, maxiter = 15)
            @test cg.value <= 1.05 * dense.value + 1e-12
            @test cg.value < 1e-2 * objective_value(obj, θ0)
            # (without a penalty, iterating past the noise level ends at different points of an
            # ill-posed valley; with the penalty the two agree)
            reg === nothing || @test norm(cg.σ - dense.σ) < 0.2 * norm(dense.σ - θ0)
            @test all(cg.σ .>= 0.1)
        end
        # the discrepancy principle stops CG-Gauss–Newton as it stops the dense one, with the
        # same quality of reconstruction
        if reg === nothing
            res = minimize(data, θ0, GaussNewton(; scaling = :sensitivity, linear_solver = :cg); lower = 0.1,
                           maxiter = 30, ftarget = target)
            ref = minimize(data, θ0, GaussNewton(; scaling = :sensitivity, linear_solver = :dense); lower = 0.1,
                           maxiter = 30, ftarget = target)
            @test res.status === :ftarget
            err(θ) = norm(conductivity(pp, θ) - σtrue)
            @test err(res.σ) < 1.2 * err(ref.σ)
            @test err(res.σ) < 0.5 * norm(conductivity(pp, θ0) - σtrue)
        end
    end

    @testset "unsupported" begin
        dobj = AdjointStateObjective(fm, trigonometric_patterns(fm, 2),
                                     forward_dirichlet(fm, σtrue, trigonometric_patterns(fm, 2))[1]; mode = :dirichlet)
        @test_throws ArgumentError jacobian_operator(dobj, ones(ndofs_σ(disc)))
        @test_throws ArgumentError GaussNewton(; linear_solver = :foo)
    end
end
