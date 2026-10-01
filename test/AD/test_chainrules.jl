# Differentiation rules (ModularEITChainRulesCoreExt): reverse rules for objective values and
# residuals of least-squares objectives, forward rules for residuals, checked against finite
# differences by ChainRulesTestUtils.
using ModularEIT
using ModularEITFerrite
using Ferrite: generate_grid, Triangle
using ChainRulesCore
using ChainRulesTestUtils
using LinearAlgebra
using Random
using Test

@testset "ChainRules" begin
    rng = MersenneTwister(4)
    disc = FerriteDiscretization(generate_grid(Triangle, (6, 6)))
    fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 8), 0.1))
    I = trigonometric_patterns(fm, 2)
    σtrue = 1 .+ rand(rng, ndofs_σ(disc))
    data, _ = forward_neumann(fm, σtrue, I)
    obj = AdjointStateObjective(fm, I, data)
    σ = 1 .+ rand(rng, ndofs_σ(disc))
    tol = (; rtol = 1e-6, atol = 1e-8, check_inferred = false)   # (the Jacobian representation is chosen at run time)

    @testset "objective value" begin
        test_rrule(objective_value, obj ⊢ NoTangent(), σ; tol...)
        reg = RegularizedObjective(obj, 1e-3 => TotalVariationRegularizer(disc; ε = 0.1))
        test_rrule(objective_value, reg ⊢ NoTangent(), σ; tol...)
    end

    @testset "residual (forward map)" begin
        @test residual(obj, σ) ≈ residual!(zeros(n_residual(obj)), obj, σ)
        test_rrule(residual, obj ⊢ NoTangent(), σ; tol...)
        test_frule(residual, obj ⊢ NoTangent(), σ; tol...)
        # the residual of an objective with zero data is the vector of measured voltages
        zero_data = AdjointStateObjective(fm, I, zero(data))
        @test residual(zero_data, σtrue) ≈ vec(data)
    end

    @testset "parametrized objective" begin
        basis = [ones(16) 0.1 .* randn(rng, 16, 2)]                     # σ ≈ θ₁ + small variations
        par = SubspaceParametrization(PixelParametrization(disc, 4, 4), basis)
        pobj = ParametrizedObjective(obj, par)
        θ = [1.0, 0.3, -0.2]
        test_rrule(objective_value, pobj ⊢ NoTangent(), θ; tol...)
        test_rrule(residual, pobj ⊢ NoTangent(), θ; tol...)
    end

    @testset "L² gradients are rejected" begin
        objL2 = AdjointStateObjective(fm, I, data; gradient = L2Gradient(FEMatrices(disc)))
        @test_throws ArgumentError ChainRulesCore.rrule(objective_value, objL2, σ)
    end
end
