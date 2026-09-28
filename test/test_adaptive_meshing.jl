# Adaptive meshing: quadrilateral meshes refined with Ferrite's AMR (hanging nodes, conformity
# constraints condensed into the u space), error indicators, marking and conductivity transfer.
using ModularEIT
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

@testset "adaptive meshing" begin
    rng = MersenneTwister(21)

    am = AdaptiveMesh(generate_grid(Quadrilateral, (4, 4)); maxlevel = 6)
    @test getncells(current_grid(am)) == 16
    amu = AdaptiveMesh(generate_grid(Quadrilateral, (2, 2)); maxlevel = 6)
    refine_mesh!(amu, [1])
    refine_mesh!(amu)                                   # every cell once: 7 cells → 28
    @test getncells(current_grid(amu)) == 28
    # levels in cell order: cell areas are 4^-level of the base cell area (1)
    lv = cell_levels(amu)
    gu = current_grid(amu)
    area(c) = (x = [get_node_coordinate(gu, n) for n in getcells(gu, c).nodes];
               abs((x[3][1] - x[1][1]) * (x[3][2] - x[1][2])))
    @test all(area(c) ≈ 4.0^-lv[c] for c in 1:getncells(gu))
    @test max_level(amu) == 6
    refine_mesh!(am, [1, 6, 7])
    grid = current_grid(am)
    @test getncells(grid) >= 16 + 3 * 3
    @test is_nonconforming(grid)

    @testset "discretization with hanging nodes" begin
        disc = FerriteDiscretization(grid)
        nhang = length(grid.conformity_info)
        @test nhang > 0
        @test ndofs_u(disc) == getnnodes(grid) - nhang
        @test ndofs_σ(disc) == getncells(grid)
        mats = FEMatrices(disc)
        nu = ndofs_u(disc)
        @test ones(nu)' * mats.M_u * ones(nu) ≈ 4.0
        @test ones(nu)' * mats.M_Γ * ones(nu) ≈ 8.0
        @test norm(mats.K_u * ones(nu)) < 1e-12
        x1 = interpolate_function(disc, x -> x[1]; field = :u)
        @test length(x1) == nu
        @test x1' * mats.K_u * x1 ≈ 4.0                 # linear functions lie in the constrained space
        @test l2_project(disc, x -> x[1]; field = :u) ≈ x1
        # boundary facets: 4 sides of length 2
        @test sum(f -> ModularEIT._facet_measure(disc, f), disc.boundary_facets) ≈ 8.0
        # conductivity tensor on the condensed pattern equals the condensed element loop
        ct = ConductivityTensor(disc)
        σ = 0.5 .+ rand(rng, ndofs_σ(disc))
        L = copy(ct.pattern)
        assemble_weighted_stiffness!(L, ct, σ)
        @test L ≈ assemble_weighted_stiffness(disc, σ)
        # TV of a step across x₁ = 0 (the refined cells straddle no interface): height 2
        step = interpolate_function(disc, x -> x[1] > 0 ? 1.0 : 0.0)
        @test total_variation(disc, step) ≈ 2.0
        # continuous σ on a non-conforming mesh is not supported
        @test_throws ArgumentError FerriteDiscretization(grid; ip_σ = Lagrange{RefQuadrilateral, 1}())
    end

    @testset "forward model and objectives on a refined mesh" begin
        disc = FerriteDiscretization(grid)
        # u = x₁ lies in the constrained space: the voltage-driven solve with boundary values x₁
        # reproduces it exactly, also at the hanging nodes
        fm = ForwardModel(disc, ContinuumModel())
        bx, _ = ModularEIT._dof_coordinates(disc, disc.boundary_dofs)
        _, X = forward_dirichlet(fm, ones(ndofs_σ(disc)), bx)
        @test X ≈ interpolate_function(disc, x -> x[1]; field = :u) atol = 1e-12

        els = angular_electrodes(disc, 4; coverage = 0.5)     # coarse mesh: few, wide electrodes
        fm = ForwardModel(disc, CompleteElectrodeModel(els, 0.1))
        σtrue = 1 .+ rand(rng, ndofs_σ(disc))
        I = trigonometric_patterns(fm, 2)
        U, _ = forward_neumann(fm, σtrue, I)
        @test dot(I[:, 2], U[:, 1]) ≈ dot(I[:, 1], U[:, 2])
        @test forward_dirichlet(fm, σtrue, U)[1] ≈ I rtol = 1e-8
        for obj in (AdjointStateObjective(fm, I, U), KohnVogeliusObjective(fm, I, U))
            σ = 1 .+ rand(rng, ndofs_σ(disc))
            gr = zeros(length(σ))
            value_and_gradient!(gr, obj, σ)
            δ = randn(rng, length(σ))
            h = 1e-5
            fd = (objective_value(obj, σ .+ h .* δ) - objective_value(obj, σ .- h .* δ)) / 2h
            @test dot(gr, δ) ≈ fd rtol = 1e-6
        end
    end

    @testset "indicators and marking" begin
        disc = FerriteDiscretization(grid)
        fm = ForwardModel(disc, ContinuumModel())
        bx, _ = ModularEIT._dof_coordinates(disc, disc.boundary_dofs)
        σ1 = ones(ndofs_σ(disc))
        _, X = forward_dirichlet(fm, σ1, bx)                 # exact solution u = x₁
        η = flux_recovery_indicator(disc, σ1, X)
        @test length(η) == getncells(grid)
        @test all(>=(0), η)
        @test sum(η) < 1e-20                               # exact linear solution: no error
        # a solution with a kink across a conductivity jump is flagged at the interface
        σjump = interpolate_function(disc, x -> x[1] > 0 ? 5.0 : 1.0)
        _, Xj = forward_dirichlet(fm, σjump, bx)
        ηj = flux_recovery_indicator(disc, σjump, Xj)
        @test sum(ηj) > 0

        J = jump_indicator(disc, σjump)
        @test length(J) == getncells(grid)
        @test all(iszero, jump_indicator(disc, σ1))
        cellmid = [sum(get_node_coordinate(grid, n) for n in getcells(grid, c).nodes) / 4 for c in 1:getncells(grid)]
        # σjump (sampled at centroids) jumps across the mesh line x₁ = 0: only cells touching it
        @test any(>(0), J)
        @test all(abs(cellmid[c][1]) < 0.3 for c in 1:getncells(grid) if J[c] > 0)
        # goal-oriented indicator (voltages of a gap model): nonnegative, zero for the exact solution
        fmg = ForwardModel(disc, GapModel(angular_electrodes(disc, 4)))
        _, Xg = forward_neumann(fmg, σjump, trigonometric_patterns(fmg, 2))
        ηg = goal_oriented_indicator(disc, fmg, σjump, Xg)
        @test length(ηg) == getncells(grid) && all(>=(0), ηg) && sum(ηg) > 0
        @test sum(goal_oriented_indicator(disc, fm, σ1, X)) < 1e-12
        θ = 0.5
        marked = dorfler_marking(ηj, θ)
        @test sum(ηj[marked]) >= θ * sum(ηj)
        @test sum(ηj[marked[1:end-1]]) < θ * sum(ηj)   # minimal
    end

    @testset "electrode positions and patterns do not depend on the mesh" begin
        coarse = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        am4 = AdaptiveMesh(generate_grid(Quadrilateral, (16, 16)); maxlevel = 6)
        refine_mesh!(am4, [2, 3, 6, 11, 40])                 # boundary cells, under electrodes
        fine = FerriteDiscretization(current_grid(am4))
        for d in (coarse, fine)
            @test ModularEIT._centroid(d, d.boundary_facets) ≈ Vec(0.0, 0.0) atol = 1e-14
        end
        # the same physical electrodes (the four sides) on both meshes: same angles, same patterns
        sides(d) = [collect(getfacetset(d.grid, n)) for n in ("right", "top", "left", "bottom")]
        fmc = ForwardModel(coarse, CompleteElectrodeModel(sides(coarse), 0.1))
        fmf = ForwardModel(fine, CompleteElectrodeModel(sides(fine), 0.1))
        @test fmc.angles ≈ fmf.angles
        @test fmc.angles ≈ [0, π / 2, π, -π / 2]
        @test trigonometric_patterns(fmc, 1) ≈ trigonometric_patterns(fmf, 1)
    end

    @testset "conductivity transfer after refinement" begin
        am2 = AdaptiveMesh(generate_grid(Quadrilateral, (4, 4)); maxlevel = 6)
        g_old = current_grid(am2)
        d_old = FerriteDiscretization(g_old)
        σ_old = rand(rng, ndofs_σ(d_old))
        refine_mesh!(am2, [2, 11])
        d_new = FerriteDiscretization(current_grid(am2))
        σ_new = transfer_conductivity(d_old, σ_old, d_new)
        @test length(σ_new) == ndofs_σ(d_new)
        # piecewise constants are transferred exactly: same integral, same values set
        @test sum(FEMatrices(d_new).M_σ * σ_new) ≈ sum(FEMatrices(d_old).M_σ * σ_old)
        @test Set(round.(σ_new; digits = 12)) == Set(round.(σ_old; digits = 12))
        # coarsening back: every parent gets the mean of its four children
        d_fine = FerriteDiscretization(current_grid(am2))
        σ_fine = rand(rng, ndofs_σ(d_fine))
        fine_cells = findall(c -> getncells(current_grid(am2)) > 0 &&
                                  ModularEIT._facet_measure(d_fine, FacetIndex(c, 1)) < 0.4, 1:getncells(current_grid(am2)))
        coarsen_mesh!(am2, fine_cells)
        d_coarse = FerriteDiscretization(current_grid(am2))
        @test getncells(current_grid(am2)) == 16
        σ_coarse = transfer_conductivity(d_fine, σ_fine, d_coarse)
        @test sum(FEMatrices(d_coarse).M_σ * σ_coarse) ≈ sum(FEMatrices(d_fine).M_σ * σ_fine)
    end

    @testset "adaptive loop reduces the estimated error" begin
        am3 = AdaptiveMesh(generate_grid(Quadrilateral, (6, 6)); maxlevel = 8)
        σf = x -> hypot(x[1] - 0.3, x[2] - 0.2) < 0.35 ? 5.0 : 1.0
        est = Float64[]
        for step in 1:3
            disc = FerriteDiscretization(current_grid(am3))
            fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 8), 0.05))
            σ = interpolate_function(disc, σf)
            _, X = forward_neumann(fm, σ, trigonometric_patterns(fm, 2))
            η = flux_recovery_indicator(disc, σ, X)
            push!(est, sqrt(sum(η)))
            refine_mesh!(am3, dorfler_marking(η, 0.5))
        end
        @test issorted(est; rev = true)
    end
end
