# The Gridap back end on Gridap's own models, and a reconstruction with it.
using ModularEIT
using ModularEITGridap
using Gridap: CartesianDiscreteModel, simplexify
using LinearAlgebra
using Random
using Test

@testset "Gridap models" begin
    for (name, model) in (("Cartesian quadrilaterals", CartesianDiscreteModel((-1, 1, -1, 1), (12, 12))),
                          ("simplexified", simplexify(CartesianDiscreteModel((-1, 1, -1, 1), (10, 10)))))
        @testset "$name" begin
            d = GridapDiscretization(model)
            mats = FEMatrices(d)
            @test sum(mats.M_σ) ≈ 4                                     # area
            @test sum(mats.M_Γ) ≈ 8                                     # perimeter
            @test electrode_length(d, ModularEIT._boundary_facets(d)) ≈ 8
            x = interpolate_function(d, p -> p[1]; field = :u)
            @test x' * mats.K_u * x ≈ 4                                 # ∫|∇x|²
            @test fe_norm(d, ones(ndofs_σ(d))) ≈ 2
            @test l2_project(d, p -> p[1] + p[2]) ≈ interpolate_function(d, p -> p[1] + p[2]) atol = 0.1
        end
    end

    @testset "reconstruction" begin
        rng = MersenneTwister(1)
        d = GridapDiscretization(simplexify(CartesianDiscreteModel((-1, 1, -1, 1), (12, 12))))
        fm = ForwardModel(d, CompleteElectrodeModel(angular_electrodes(d, 16), 0.05))
        phantom = InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.35, 3.0)])
        σtrue = conductivity(d, phantom)
        I = trigonometric_patterns(fm, 6)
        U, _ = forward_neumann(fm, σtrue, I)
        obj = RegularizedObjective(AdjointStateObjective(fm, I, U), 1e-5 => TotalVariationRegularizer(d; ε = 1e-2))
        σ0 = ones(ndofs_σ(d))
        res = minimize(obj, σ0, GaussNewton(); lower = 0.05, maxiter = 15)
        err(σ) = norm(σ - σtrue) / norm(σtrue .- 1)
        @test objective_value(obj, res.σ) < 1e-2 * objective_value(obj, σ0)
        @test err(res.σ) < 0.7 * err(σ0)
    end
end
