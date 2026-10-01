# EITProblem: a reconstruction problem as one object (discretization, forward model, data,
# objective, current iterate), solved and warm-started by reconstruct!.
using ModularEIT
using ModularEITFerrite
using Ferrite: generate_grid, Triangle
using LinearAlgebra
using Random
using Test

@testset "EITProblem" begin
    rng = MersenneTwister(7)
    disc = FerriteDiscretization(generate_grid(Triangle, (12, 12)))
    fine = FerriteDiscretization(generate_grid(Triangle, (24, 24)))
    els = angular_electrodes(disc, 16)
    model = CompleteElectrodeModel(els, 0.05)
    fm_fine = ForwardModel(fine, CompleteElectrodeModel(transfer_electrodes(disc, els, fine), 0.05))
    phantom = InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.4, 2.5)])
    currents = trigonometric_patterns(ForwardModel(disc, model), 6)
    noise = RelativeGaussianNoise(0.01)
    sim = simulate_data(fine, fm_fine, phantom, currents; noise, rng)
    truth = conductivity(disc, phantom)
    err(σ) = norm(σ - truth) / norm(truth .- 1)

    @testset "construction and delegation" begin
        prob = EITProblem(disc, model, currents, sim.data; noise)
        @test prob isa AbstractEITProblem
        @test solution(prob) == ones(ndofs_σ(disc))
        @test prob.forward isa ForwardModel
        @test prob.target ≈ discrepancy_target(prob.data, noise)
        @test objective_value(prob) ≈ objective_value(prob.objective, solution(prob))
        g1, g2 = zeros(ndofs_σ(disc)), zeros(ndofs_σ(disc))
        σ = 1 .+ 0.1 .* rand(rng, ndofs_σ(disc))
        @test value_and_gradient!(g1, prob, σ) ≈ value_and_gradient!(g2, prob.objective, σ)
        @test g1 ≈ g2
        @test data_misfit(prob) ≈ objective_value(prob.data, solution(prob))
        # a forward model can be passed instead of an electrode model
        p2 = EITProblem(disc, prob.forward, currents, sim.data; initial = 2.0)
        @test p2.forward === prob.forward
        @test solution(p2) == fill(2.0, ndofs_σ(disc))
        @test occursin("EITProblem", sprint(show, prob))
    end

    @testset "discrepancy principle and warm start" begin
        prob = EITProblem(disc, model, currents, sim.data; noise)
        st = reconstruct!(prob, GaussNewton(); maxiter = 30)
        @test st.status === :ftarget
        @test data_misfit(prob) <= prob.target
        @test err(solution(prob)) < 0.8
        @test length(prob.history) == 1
        # warm start: already at the target, so the next run stops at once
        st2 = reconstruct!(prob, GaussNewton(); maxiter = 30)
        @test st2.iteration == 0
        @test length(prob.history) == 2
    end

    @testset "regularization: discrepancy on the data misfit" begin
        prob = EITProblem(disc, model, currents, sim.data; noise,
                          regularization = (1e-4 => TotalVariationRegularizer(disc; ε = 1e-2),))
        @test prob.objective isa RegularizedObjective
        st = reconstruct!(prob, GaussNewton(); maxiter = 40)
        @test data_misfit(prob) <= prob.target
        @test err(solution(prob)) < 0.8
        # without a noise model there is no target: runs to maxiter or convergence
        p2 = EITProblem(disc, model, currents, sim.data;
                        regularization = (1e-4 => TikhonovRegularizer(disc; kind = :jump),))
        @test isnan(p2.target)
        st = reconstruct!(p2, LBFGS(); maxiter = 5)
        @test st.iteration == 5
        @test objective_value(p2) < objective_value(p2.objective, ones(ndofs_σ(disc)))
    end

    @testset "proximal methods get the data term" begin
        tv = TotalVariationRegularizer(disc; ε = 0.0)
        prob = EITProblem(disc, model, currents, sim.data; noise)
        st = reconstruct!(prob, ProximalGradient(1e-4 => tv; weights = lumped_mass(disc)); maxiter = 15)
        @test objective_value(prob.data, solution(prob)) < objective_value(prob.data, ones(ndofs_σ(disc)))
    end

    @testset "parametrization" begin
        pp = PixelParametrization(disc, 8, 8)
        prob = EITProblem(disc, model, currents, sim.data; noise, parametrization = pp)
        @test length(prob.θ) == 64
        @test solution(prob) ≈ conductivity(pp, prob.θ)
        reconstruct!(prob, GaussNewton(); maxiter = 30)
        @test data_misfit(prob) < objective_value(prob.data, ones(64))
        @test err(solution(prob)) < 0.9
    end

    @testset "input checks" begin
        @test_throws DimensionMismatch EITProblem(disc, model, currents, sim.data; initial = ones(3))
    end
end
