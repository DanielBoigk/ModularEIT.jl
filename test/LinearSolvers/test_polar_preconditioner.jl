# Fast solver on rotationally symmetric disk meshes: mesh generator, detection of the polar
# structure, the FFT-in-θ pseudo-inverse, exactness for constant σ, iteration bounds for variable σ
# and agreement with the direct solver.
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

disk(nr, nθ; kwargs...) = FerriteDiscretization(polar_grid(nr, nθ; kwargs...))

disk_models(disc) = (
    ("continuum", ContinuumModel()),
    ("point", PointElectrodeModel([Vec(cos(t), sin(t)) for t in (0.1, 1.7, 3.3, 4.9)])),
    ("gap", GapModel(angular_electrodes(disc, 8; coverage = 0.5))),
    ("CEM", CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.05)),
)

function polar_solve(disc, fm, σ, B; system = :neumann, rtol = 1e-10)
    system_matrix!(fm, σ)
    A = system === :neumann ? fm.A : fm.A_ff
    M = polar_preconditioner(disc, fm; system)
    ModularEIT.update_preconditioner!(M, A)
    k = size(A, 1)
    ws = BlockCGWorkspace(A, B; nullspace = system === :neumann ? fm.nullspace : zeros(k, 0),
                          grounding = system === :neumann ? fm.grounding : zeros(k, 0))
    X = zeros(size(B))
    return X, pbcg!(X, ws, A, B; M, rtol)
end

@testset "polar preconditioner" begin
    rng = MersenneTwister(5)

    @testset "mesh generator" begin
        grid = polar_grid(4, 12; radius = 2.0)
        @test getnnodes(grid) == 1 + 4 * 12
        @test getncells(grid) == 12 + 2 * 12 * 3
        r = [norm(n.x) for n in grid.nodes]
        @test maximum(r) ≈ 2.0
        disc = FerriteDiscretization(grid)
        mats = FEMatrices(disc)
        @test sum(mats.M_σ) ≈ 12 / 2 * 2.0^2 * sin(2π / 12)       # area of the 12-gon
        # graded rings: the outermost spacing is prescribed
        g = polar_grid(10, 64; boundary_spacing = 0.02)
        radii = sort(unique(round.([norm(n.x) for n in g.nodes]; digits = 12)))
        @test length(radii) == 11 && radii[end] ≈ 1
        @test radii[end] - radii[end - 1] ≈ 0.02 rtol = 1e-8
        @test issorted(diff(radii); rev = true)                   # spacing decreases outwards
        @test_throws ArgumentError polar_grid(10, 64; boundary_spacing = 0.5)  # would need growing spacing
    end

    @testset "structure detection" begin
        disc = disk(5, 16)
        p = ModularEIT.polar_structure(disc)
        @test (p.nr, p.nθ) == (5, 16)
        @test sort(p.perm) == 1:ndofs_u(disc)
        @test_throws ArgumentError ModularEIT.polar_structure(FerriteDiscretization(generate_grid(Triangle, (4, 4))))
        # a mesh that is not rotationally symmetric is rejected
        grid = polar_grid(5, 16)
        n = grid.nodes[10]
        grid.nodes[10] = Node(n.x * 1.01)
        @test_throws ArgumentError ModularEIT.polar_structure(FerriteDiscretization(grid))
    end

    @testset "FFT solver = pseudo-inverse of the P1 stiffness matrix" begin
        disc = disk(6, 20; boundary_spacing = 0.08)
        p = ModularEIT.polar_structure(disc)
        K = assemble_stiffness(disc.dh_u, disc.cv_u)[p.perm, p.perm]      # structure order
        b = randn(rng, size(K, 1), 3)
        b .-= sum(b; dims = 1) ./ size(b, 1)
        y = ModularEIT.fast_neumann_solve(p, b)
        @test K * y ≈ b
        @test norm(sum(y; dims = 1)) < 1e-10 * norm(y)                    # Moore–Penrose: mean zero
        c = randn(rng, size(K, 1))
        c .-= sum(c) / length(c)
        @test dot(c, ModularEIT.fast_neumann_solve(p, b[:, 1])) ≈ dot(b[:, 1], ModularEIT.fast_neumann_solve(p, c))
    end

    @testset "exact for constant σ: $name ($system)" for (name, model) in disk_models(disk(8, 48)),
                                                        system in (:neumann, :dirichlet)
        disc = disk(8, 48)
        fm = ForwardModel(disc, model)
        σ = fill(0.7, ndofs_σ(disc))
        if system === :neumann
            B = Matrix(fm.P * trigonometric_patterns(fm, 2))
        else
            system_matrix!(fm, σ)
            B = randn(rng, length(fm.free_dofs), 3)
        end
        _, stats = polar_solve(disc, fm, σ, B; system)
        @test stats.converged
        @test stats.iterations <= 2
    end

    @testset "iterations bounded by the contrast" begin
        for (nr, nθ) in ((8, 64), (16, 128), (32, 256))                     # nθ = 4L: edges on nodes
            disc = disk(nr, nθ; boundary_spacing = 1 / (2nr))           # half the uniform spacing
            fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.05))
            σ = interpolate_function(disc, x -> 1 + 9 * (norm(x - Vec(0.3, 0.2)) < 0.35))
            B = Matrix(fm.P * trigonometric_patterns(fm, 3))
            _, stats = polar_solve(disc, fm, σ, B)
            @test stats.converged
            @test stats.iterations <= ceil(sqrt(10) * log(2 / 1e-10) / 2)
        end
    end

    @testset "objectives: same results as the direct solver" begin
        disc = disk(10, 40; boundary_spacing = 0.1)
        σtrue = interpolate_function(disc, x -> 1 + (norm(x - Vec(0.3, 0.1)) < 0.4))
        σ = 1 .+ 0.3 .* rand(rng, ndofs_σ(disc))
        solver = BlockCGSolver(; preconditioner = PolarPreconditioner(disc), rtol = 1e-13)
        for model in (ContinuumModel(), CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.1))
            fm = ForwardModel(disc, model)
            for mode in (:neumann, :dirichlet)
                inputs = trigonometric_patterns(fm, 2)
                obs = mode === :neumann ? forward_neumann(fm, σtrue, inputs)[1] : forward_dirichlet(fm, σtrue, inputs)[1]
                gd, gc = zeros(length(σ)), zeros(length(σ))
                Jd = value_and_gradient!(gd, AdjointStateObjective(fm, inputs, obs; mode), σ)
                Jc = value_and_gradient!(gc, AdjointStateObjective(fm, inputs, obs; mode, solver), σ)
                @test Jc ≈ Jd rtol = 1e-8
                @test gc ≈ gd rtol = 1e-7
            end
        end
    end
end
