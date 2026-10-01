# DCT preconditioner on uniform rectangle grids: grid detection, the transform solver, exactness
# for constant σ (all electrode models, current- and voltage-driven), mesh-independent iteration
# counts for variable σ, and agreement with the direct solver inside objectives.
using ModularEIT
using ModularEITFerrite
using Ferrite
using SparseArrays
using LinearAlgebra
using Random
using Test

rect(nx, ny; lo = (-1.0, -1.0), hi = (1.0, 1.0)) =
    FerriteDiscretization(generate_grid(Quadrilateral, (nx, ny), Vec(lo...), Vec(hi...)))

models(disc) = (
    ("continuum", ContinuumModel()),
    ("point", PointElectrodeModel([Vec(1.0, 0.0), Vec(0.0, 1.0), Vec(-1.0, 0.0), Vec(0.0, -1.0)])),
    ("gap", GapModel(angular_electrodes(disc, 8; coverage = 0.5))),
    ("CEM", CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.05)),
)

# pbcg with the DCT preconditioner on the system of a forward model; returns (X, stats)
function dct_solve(disc, fm, σ, B; system = :neumann, variant = :constant, rtol = 1e-10)
    system_matrix!(fm, σ)
    A = system === :neumann ? fm.A : fm.A_ff
    M = dct_preconditioner(disc, fm; system, variant)
    ModularEIT.update_preconditioner!(M, A)
    k = system === :neumann ? 1 : 0
    ws = BlockCGWorkspace(A, B; nullspace = system === :neumann ? fm.nullspace : zeros(size(A, 1), 0),
                          grounding = system === :neumann ? fm.grounding : zeros(size(A, 1), 0))
    X = zeros(size(B))
    stats = pbcg!(X, ws, A, B; M, rtol)
    return X, stats
end

@testset "DCT preconditioner" begin
    rng = MersenneTwister(3)

    @testset "grid detection" begin
        g = ModularEIT.structured_grid(rect(6, 4; lo = (0.0, 0.0), hi = (3.0, 1.0)))
        @test (g.nx, g.ny) == (7, 5)                               # nodes
        @test g.hx ≈ 0.5 && g.hy ≈ 0.25
        @test sort(g.perm) == 1:35                                  # a permutation of the u dofs
        # lexicographic order: x fastest
        disc = rect(6, 4; lo = (0.0, 0.0), hi = (3.0, 1.0))
        xy = ModularEITFerrite._dof_coordinates(disc, g.perm)
        @test xy[1][1:3] ≈ [0.0, 0.5, 1.0] && xy[2][1:3] ≈ [0.0, 0.0, 0.0]
        @test xy[2][8] ≈ 0.25
        @test_throws ArgumentError ModularEIT.structured_grid(FerriteDiscretization(generate_grid(Triangle, (4, 4))))
        @test_throws ArgumentError ModularEIT.structured_grid(FerriteDiscretization(generate_grid(Quadrilateral, (4, 4));
                                                                                    ip_u = Lagrange{RefQuadrilateral, 2}()))
        # distorted nodes are rejected
        grid = generate_grid(Quadrilateral, (4, 4))
        n = grid.nodes[7]
        grid.nodes[7] = Node(n.x + Vec(0.05, 0.0))
        @test_throws ArgumentError ModularEIT.structured_grid(FerriteDiscretization(grid))
    end

    @testset "transform solver = pseudo-inverse of the Q1 stiffness matrix" begin
        disc = rect(9, 6; lo = (0.0, 0.0), hi = (2.0, 0.5))
        g = ModularEIT.structured_grid(disc)
        K = assemble_stiffness(disc.dh_u, disc.cv_u)[g.perm, g.perm]      # lexicographic order
        b = randn(rng, size(K, 1), 3)
        b .-= sum(b; dims = 1) ./ size(b, 1)                              # range of K
        y = ModularEIT.dct_neumann_solve(g, b)
        @test K * y ≈ b
        w = vec(g.wx * g.wy')                                             # trapezoid weights
        @test norm(w' * y) < 1e-10 * norm(y)                              # grounding: ∫ u = 0
    end

    @testset "exact for constant σ: $name ($system)" for (name, model) in models(rect(16, 16)),
                                                        system in (:neumann, :dirichlet)
        disc = rect(16, 16)
        fm = ForwardModel(disc, model)
        σ = fill(2.5, ndofs_σ(disc))
        if system === :neumann
            I_ = trigonometric_patterns(fm, 2)
            B = Matrix(fm.P * I_)
        else
            n_control(fm) > 0 || continue
            system_matrix!(fm, σ)
            B = randn(rng, length(fm.free_dofs), 3)
        end
        X, stats = dct_solve(disc, fm, σ, B; system)
        @test stats.converged
        @test stats.iterations <= 2
    end

    # :constant for a jump (contrast 10); :scaled for a smooth σ (at jumps its condition number
    # grows with refinement, like the Concus–Golub transformation Δ√σ/√σ)
    @testset "mesh-independent iteration counts ($variant)" for (variant, f) in
            ((:constant, x -> 1 + 9 * (norm(x - Vec(0.2, 0.1)) < 0.4)), (:scaled, x -> exp(2 * x[1] + x[2])))
        its = map((16, 32, 64)) do N
            disc = rect(N, N)
            fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.05))
            σ = interpolate_function(disc, f)
            B = Matrix(fm.P * trigonometric_patterns(fm, 3))
            _, stats = dct_solve(disc, fm, σ, B; variant)
            @test stats.converged
            stats.iterations
        end
        # κ ≤ contrast (constant variant), so CG needs at most ⌈√κ ln(2/rtol) / 2⌉ iterations on
        # every mesh; coarse meshes converge earlier (fewer distinct eigenvalues)
        @test maximum(its) <= (variant === :constant ? ceil(sqrt(10) * log(2 / 1e-10) / 2) : 60)
    end

    @testset "scaled variant for smooth high-contrast σ" begin
        disc = rect(32, 32)
        fm = ForwardModel(disc, ContinuumModel())
        σ = interpolate_function(disc, x -> exp(2 * x[1] + x[2]))       # contrast e⁶ ≈ 400
        B = Matrix(fm.P * trigonometric_patterns(fm, 2))
        _, sc = dct_solve(disc, fm, σ, B; variant = :constant)
        _, ss = dct_solve(disc, fm, σ, B; variant = :scaled)
        @test sc.converged && ss.converged
        @test ss.iterations < sc.iterations
    end

    @testset "objectives: same results as the direct solver" begin
        disc = rect(20, 12; lo = (0.0, 0.0), hi = (2.0, 1.0))
        σtrue = interpolate_function(disc, x -> 1 + (norm(x - Vec(1.2, 0.5)) < 0.3))
        σ = 1 .+ 0.3 .* rand(rng, ndofs_σ(disc))
        solver = BlockCGSolver(; preconditioner = DCTPreconditioner(disc), rtol = 1e-13)
        for (name, model) in (("continuum", ContinuumModel()),
                              ("CEM", CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.1)))
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
        @test_throws ArgumentError BlockCGSolver(; preconditioner = :dct)       # needs the discretization
    end
end
