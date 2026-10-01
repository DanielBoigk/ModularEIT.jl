# Residual-based error indicator: element residuals, current-density jumps across interior
# facets (conforming and hanging), boundary residuals of every electrode model.
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

@testset "residual estimator" begin
    rng = MersenneTwister(13)

    @testset "exact linear solutions give zero (conforming, triangles, hanging nodes)" begin
        am = AdaptiveMesh(generate_grid(Quadrilateral, (4, 4)); maxlevel = 5)
        refine_mesh!(am, [1, 6, 7])
        amt = AdaptiveMesh(generate_grid(Triangle, (4, 4)))
        refine_mesh!(amt, [2, 9, 10])
        for grid in (generate_grid(Triangle, (5, 5)), current_grid(am), current_grid(amt))
            disc = FerriteDiscretization(grid)
            fm = ForwardModel(disc, ContinuumModel())
            bx, by = ModularEITFerrite._dof_coordinates(disc, disc.boundary_dofs)
            σ = ones(ndofs_σ(disc))
            _, X = forward_dirichlet(fm, σ, bx .+ 2 .* by)          # u = x₁ + 2x₂
            η = residual_indicator(disc, fm, σ, X; mode = :dirichlet)
            @test length(η) == getncells(grid)
            @test sum(η) < 1e-20
        end
    end

    @testset "flux jumps are measured with the conductivity" begin
        # u piecewise linear with a kink at x₁ = 0 and σ jumping there such that σ∂₁u is
        # continuous: the discrete solution is exact, the indicator vanishes. With the wrong
        # σ ratio the jump is flagged on the interface cells only.
        grid = generate_grid(Quadrilateral, (4, 4))
        disc = FerriteDiscretization(grid)
        fm = ForwardModel(disc, ContinuumModel())
        bx, by = ModularEITFerrite._dof_coordinates(disc, disc.boundary_dofs)
        f = [x < 0 ? 2x : x for x in bx]                     # ∂₁u = 2 left, 1 right
        σ = interpolate_function(disc, x -> x[1] < 0 ? 1.0 : 2.0)
        _, X = forward_dirichlet(fm, σ, f)
        @test sum(residual_indicator(disc, fm, σ, X; mode = :dirichlet)) < 1e-20
        σbad = interpolate_function(disc, x -> x[1] < 0 ? 1.0 : 3.0)
        _, Xb = forward_dirichlet(fm, σbad, f)
        ηb = residual_indicator(disc, fm, σbad, Xb; mode = :dirichlet)
        @test sum(ηb) > 0
    end

    @testset "O(h) under uniform refinement (smooth problem)" begin
        est = Float64[]
        for m in (8, 16, 32)
            disc = FerriteDiscretization(generate_grid(Triangle, (m, m)))
            fm = ForwardModel(disc, ContinuumModel())
            I = trigonometric_patterns(fm, 1)
            σ = interpolate_function(disc, x -> 1 + 0.5 * x[1]^2; field = :σ)
            _, X = forward_neumann(fm, σ, I)
            push!(est, sqrt(sum(residual_indicator(disc, fm, σ, X, I))))
        end
        @test 1.6 < est[1] / est[2] < 2.6
        @test 1.6 < est[2] / est[3] < 2.6
    end

    @testset "boundary residuals of the electrode models" begin
        grid = generate_grid(Quadrilateral, (12, 12))
        disc = FerriteDiscretization(grid)
        els = angular_electrodes(disc, 8)
        σ = 1 .+ rand(rng, ndofs_σ(disc))
        for model in (ContinuumModel(), GapModel(els), CompleteElectrodeModel(els, 0.05),
                      PointElectrodeModel([Vec(cos(t), sin(t)) for t in 2π .* (0:7) ./ 8]))
            fm = ForwardModel(disc, model)
            I = trigonometric_patterns(fm, 2)
            _, X = forward_neumann(fm, σ, I)
            η = residual_indicator(disc, fm, σ, X, I)
            @test length(η) == getncells(grid) && all(>=(0), η) && sum(η) > 0
            # for σ ≡ 1 there are no conductivity jumps: the largest indicators sit at the
            # boundary (electrode edges), where the boundary residual lives
            if !(model isa PointElectrodeModel)
                s1 = ones(ndofs_σ(disc))
                _, X1 = forward_neumann(fm, s1, I)
                η1 = residual_indicator(disc, fm, s1, X1, I)
                bcells = Set(first(f.idx) for f in disc.boundary_facets)
                @test argmax(η1) in bcells
            end
        end
        # the boundary residual is consistent: refining uniformly reduces the CEM estimate
        est = Float64[]
        for m in (8, 16, 32)
            d = FerriteDiscretization(generate_grid(Quadrilateral, (m, m)))
            fm = ForwardModel(d, CompleteElectrodeModel(angular_electrodes(d, 4; coverage = 0.4), 0.5))
            I = trigonometric_patterns(fm, 1)
            s = ones(ndofs_σ(d))
            _, X = forward_neumann(fm, s, I)
            push!(est, sqrt(sum(residual_indicator(d, fm, s, X, I))))
        end
        @test est[1] > est[2] > est[3]
    end

    @testset "goal-oriented indicator with residual estimates" begin
        am = AdaptiveMesh(generate_grid(Quadrilateral, (8, 8)); maxlevel = 5)
        refine_mesh!(am, [1, 2, 3])
        disc = FerriteDiscretization(current_grid(am))
        fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 8), 0.05))
        σ = 1 .+ rand(rng, ndofs_σ(disc))
        I = trigonometric_patterns(fm, 2)
        _, X = forward_neumann(fm, σ, I)
        η = goal_oriented_indicator(disc, fm, σ, X; estimator = :residual, currents = I)
        @test length(η) == getncells(current_grid(am)) && all(>=(0), η) && sum(η) > 0
        @test_throws ArgumentError goal_oriented_indicator(disc, fm, σ, X; estimator = :residual)
    end
end
