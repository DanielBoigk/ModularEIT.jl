# Objectives: adjoint-state least squares (Neumann and Dirichlet mode) and Kohn–Vogelius.
# Gradients and Jacobians are checked against central finite differences in random directions.
using ModularEIT
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

function build_model(kind, disc; L = 8)
    els = angular_electrodes(disc, L; coverage = 0.5)
    model = kind === :continuum ? ContinuumModel() :
            kind === :point ? PointElectrodeModel([Vec(cos(t), sin(t)) for t in 2π .* (0:L-1) ./ L]) :
            kind === :gap ? GapModel(els) :
            CompleteElectrodeModel(els, 0.1)
    return ForwardModel(disc, model)
end

# synthetic data for a mode: (inputs, observations)
function synthetic_data(fm, σ, mode; K = 2)
    inputs = trigonometric_patterns(fm, K)
    obs = mode === :neumann ? forward_neumann(fm, σ, inputs)[1] : forward_dirichlet(fm, σ, inputs)[1]
    return inputs, obs
end

central_fd(f, σ, δ, h) = (f(σ .+ h .* δ) .- f(σ .- h .* δ)) ./ 2h

@testset "objectives" begin
    rng = MersenneTwister(9)
    tri = generate_grid(Triangle, (8, 8))
    quad = generate_grid(Quadrilateral, (8, 8))
    discs = [
        ("P1/P0 triangles", FerriteDiscretization(tri)),
        ("P1/P1 triangles", FerriteDiscretization(tri; ip_σ = Lagrange{RefTriangle, 1}())),
        ("Q1/Q0 quads", FerriteDiscretization(quad)),
    ]

    @testset "adjoint state: $kind, $mode, $dname" for (dname, disc) in discs,
                                                       kind in (:continuum, :point, :gap, :cem),
                                                       mode in (:neumann, :dirichlet)
        nσ = ndofs_σ(disc)
        fm = build_model(kind, disc)
        σtrue = 1 .+ 0.5 .* rand(rng, nσ)
        inputs, obs = synthetic_data(fm, σtrue, mode)
        obj = AdjointStateObjective(fm, inputs, obs; mode)
        @test obj isa AbstractObjective
        g = zeros(nσ)
        @test value_and_gradient!(g, obj, σtrue) < 1e-18
        @test norm(g) < 1e-8

        σ = 1 .+ 0.5 .* rand(rng, nσ)
        J = value_and_gradient!(g, obj, σ)
        @test J > 0
        @test objective_value(obj, σ) ≈ J
        δ = randn(rng, nσ)
        h = 1e-5
        fd = central_fd(s -> objective_value(obj, s), σ, δ, h)
        @test dot(g, δ) ≈ fd rtol = 1e-6

        # residual and Jacobian: J δ = directional derivative of the residual, Jᵀ r = gradient
        m = n_residual(obj)
        r = zeros(m)
        Jm = zeros(m, nσ)
        residual_and_jacobian!(r, Jm, obj, σ)
        @test J ≈ dot(r, r) / 2
        fdr = central_fd(s -> residual!(zeros(m), obj, s), σ, δ, h)
        @test Jm * δ ≈ fdr rtol = 1e-6
        @test Jm' * r ≈ g rtol = 1e-8
    end

    @testset "L² gradient = M_σ⁻¹ × coefficient gradient" begin
        disc = discs[1][2]
        fm = build_model(:cem, disc)
        σtrue = 1 .+ rand(rng, ndofs_σ(disc))
        inputs, obs = synthetic_data(fm, σtrue, :neumann)
        mats = FEMatrices(disc)
        obj_c = AdjointStateObjective(fm, inputs, obs)
        obj_l2 = AdjointStateObjective(fm, inputs, obs; gradient = L2Gradient(mats))
        σ = ones(ndofs_σ(disc))
        gc, gl = zeros(length(σ)), zeros(length(σ))
        @test value_and_gradient!(gc, obj_c, σ) ≈ value_and_gradient!(gl, obj_l2, σ)
        @test mats.M_σ * gl ≈ gc
    end

    @testset "weighted misfit" begin
        disc = discs[1][2]
        fm = build_model(:gap, disc)
        inputs, obs = synthetic_data(fm, 1 .+ rand(rng, ndofs_σ(disc)), :neumann)
        W = Diagonal(0.5 .+ rand(rng, n_measure(fm)))
        obj = AdjointStateObjective(fm, inputs, obs; misfit = WeightedSquaredEuclidean(W))
        σ = 1 .+ rand(rng, ndofs_σ(disc))
        g = zeros(length(σ))
        value_and_gradient!(g, obj, σ)
        δ = randn(rng, length(σ))
        @test dot(g, δ) ≈ central_fd(s -> objective_value(obj, s), σ, δ, 1e-5) rtol = 1e-6
    end

    @testset "swappable linear solvers" begin
        disc = discs[3][2]
        fm = build_model(:cem, disc)
        σtrue = 1 .+ rand(rng, ndofs_σ(disc))
        σ = 1 .+ rand(rng, ndofs_σ(disc))
        for mode in (:neumann, :dirichlet)
            inputs, obs = synthetic_data(fm, σtrue, mode)
            gd = zeros(length(σ))
            Jd = value_and_gradient!(gd, AdjointStateObjective(fm, inputs, obs; mode), σ)
            for pc in (:amg, :jacobi)
                solver = BlockCGSolver(; preconditioner = pc, rtol = 1e-13)
                obj = AdjointStateObjective(fm, inputs, obs; mode, solver)
                gc = zeros(length(σ))
                @test value_and_gradient!(gc, obj, σ) ≈ Jd rtol = 1e-7
                @test gc ≈ gd rtol = 1e-6
            end
        end
    end

    @testset "Kohn–Vogelius: $kind, $dname" for (dname, disc) in discs, kind in (:continuum, :cem)
        nσ = ndofs_σ(disc)
        fm = build_model(kind, disc)
        σtrue = 1 .+ 0.5 .* rand(rng, nσ)
        currents = trigonometric_patterns(fm, 2)
        voltages = forward_neumann(fm, σtrue, currents)[1]
        obj = KohnVogeliusObjective(fm, currents, voltages)
        @test obj isa AbstractObjective
        g = zeros(nσ)
        @test value_and_gradient!(g, obj, σtrue) < 1e-16       # zero at the true conductivity
        @test norm(boundary_error(obj)) < 1e-8
        σ = 1 .+ 0.5 .* rand(rng, nσ)
        J = value_and_gradient!(g, obj, σ)
        @test J > 0
        @test length(pattern_values(obj)) == size(currents, 2)
        @test sum(pattern_values(obj)) ≈ J
        δ = randn(rng, nσ)
        @test dot(g, δ) ≈ central_fd(s -> objective_value(obj, s), σ, δ, 1e-5) rtol = 1e-6
    end

    @testset "Kohn–Vogelius warns for inconsistent electrode models" begin
        disc = discs[1][2]
        fm = build_model(:gap, disc)
        currents = trigonometric_patterns(fm, 2)
        voltages = forward_neumann(fm, ones(ndofs_σ(disc)), currents)[1]
        @test_logs (:warn, r"not consistent") KohnVogeliusObjective(fm, currents, voltages)
    end
end
