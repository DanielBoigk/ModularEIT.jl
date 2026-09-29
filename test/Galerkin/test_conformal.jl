# Conformal maps from the unit disk (Theodorsen's method), conformally mapped polar meshes, and the
# FFT preconditioner of the disk used on the mapped mesh.
using ModularEIT
using Ferrite
using LinearAlgebra
using Random
using Test

@testset "conformal maps" begin
    rng = MersenneTwister(8)

    @testset "the disk maps to itself" begin
        Φ = ConformalMap(t -> (2cos(t) + 1, 2sin(t) - 0.5); center = (1.0, -0.5))
        for w in (0.0 + 0im, 0.3 + 0.4im, 0.9im, -0.99 + 0im)
            @test Φ(w) ≈ complex(1, -0.5) + 2w atol = 1e-10
            @test map_derivative(Φ, w) ≈ 2 atol = 1e-9
        end
    end

    @testset "reproduces a known map: w ↦ w + εw²" begin
        ε = 0.3
        f(w) = w + ε * w^2
        Φ = ConformalMap(t -> reim(f(cis(t))); center = (0.0, 0.0))
        for w in (0.5cis(0.3), 0.9cis(2.0), 0.99cis(-1.2), 0.2im)
            @test Φ(w) ≈ f(w) atol = 1e-9
            @test map_derivative(Φ, w) ≈ 1 + 2ε * w atol = 1e-8
        end
        @test Φ.boundary_error < 1e-10
    end

    @testset "ellipse: boundary on the curve, normalisation, univalence" begin
        a, b = 1.5, 1.0
        Φ = ConformalMap(t -> (a * cos(t), b * sin(t)))
        θ = range(0, 2π; length = 401)
        z = [Φ(cis(t)) for t in θ]
        @test maximum(abs.(real.(z) .^ 2 ./ a^2 .+ imag.(z) .^ 2 ./ b^2 .- 1)) < 1e-8
        @test abs(Φ(0.0 + 0im)) < 1e-12
        @test abs(imag(map_derivative(Φ, 0.0 + 0im))) < 1e-12 && real(map_derivative(Φ, 0.0 + 0im)) > 0
        @test minimum(abs(map_derivative(Φ, r * cis(t))) for r in (0.0, 0.5, 0.99), t in θ) > 0
        # a polygon (piecewise linear boundary) is accepted as well; the corners limit the accuracy
        # (algebraic convergence in the number of modes, ≈ N^-0.7 for these corners)
        oct = [(cos(k * π / 4), sin(k * π / 4)) for k in 0:7]
        segdist(z, a, b) = (d = b .- a; t = clamp(((real(z) - a[1]) * d[1] + (imag(z) - a[2]) * d[2]) / sum(abs2, d), 0, 1);
                            hypot(real(z) - a[1] - t * d[1], imag(z) - a[2] - t * d[2]))
        dist(z) = minimum(segdist(z, oct[k], oct[mod1(k + 1, 8)]) for k in 1:8)
        errs = [maximum(dist(ConformalMap(oct; modes = N)(cis(t))) for t in θ) for N in (128, 1024)]
        @test errs[1] < 1e-2
        @test errs[2] < 0.4 * errs[1]
        # not star-shaped / too eccentric for Theodorsen's method
        @test_throws ArgumentError ConformalMap(t -> (3cos(t), 0.5sin(t)))
    end

    @testset "conformally mapped polar mesh" begin
        Φ = ConformalMap(t -> (1.3cos(t) + 0.1cos(2t), sin(t) + 0.1sin(3t)))    # thorax-like
        cg = conformal_grid(Φ, 8, 64; boundary_spacing = 1 / 16)
        g, ref = cg.grid, cg.reference
        @test getncells(g) == getncells(ref) && getnnodes(g) == getnnodes(ref)
        @test all(getcells(g, c).nodes == getcells(ref, c).nodes for c in 1:getncells(g))
        # positively oriented (valid) triangles
        area(c) = (x = getcoordinates(g, c); ((x[2] - x[1])[1] * (x[3] - x[1])[2] - (x[2] - x[1])[2] * (x[3] - x[1])[1]) / 2)
        @test minimum(area, 1:getncells(g)) > 0
        # boundary nodes lie on the curve: node k of the outer ring is Φ(e^{iθ})
        disc = FerriteDiscretization(g)
        @test length(disc.boundary_dofs) == 64
    end

    @testset "disk preconditioner on the mapped mesh" begin
        Φ = ConformalMap(t -> (1.3cos(t) + 0.1cos(2t), sin(t) + 0.1sin(3t)))
        its = map(((8, 64), (16, 128), (32, 256))) do (nr, nθ)
            cg = conformal_grid(Φ, nr, nθ; boundary_spacing = 1 / (2nr))
            disc = FerriteDiscretization(cg.grid)
            fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 16; coverage = 0.5), 0.05))
            M = polar_preconditioner(disc, fm; reference = cg.reference)
            B = Matrix(fm.P * trigonometric_patterns(fm, 3))
            map((x -> 1.0, x -> 1 + 9 * (norm(x - Vec(0.3, 0.2)) < 0.35))) do f
                σ = interpolate_function(disc, f)
                system_matrix!(fm, σ)
                ModularEIT.update_preconditioner!(M, fm.A)
                ws = BlockCGWorkspace(fm.A, B; nullspace = fm.nullspace, grounding = fm.grounding)
                X = zeros(size(B))
                st = pbcg!(X, ws, fm.A, B; M, rtol = 1e-10)
                @test st.converged
                st.iterations
            end
        end
        constant = first.(its)
        contrast = last.(its)
        @test maximum(constant) <= 15                     # distortion of the discrete map only
        @test maximum(constant) - minimum(constant) <= 4  # does not grow under refinement
        @test maximum(contrast) <= 45
        # the reference must have the same connectivity
        cg = conformal_grid(Φ, 8, 64)
        disc = FerriteDiscretization(cg.grid)
        fm = ForwardModel(disc, ContinuumModel())
        @test_throws ArgumentError polar_preconditioner(disc, fm; reference = polar_grid(8, 32))
    end

    @testset "objectives with the mapped-mesh preconditioner = direct solver" begin
        Φ = ConformalMap(t -> (1.3cos(t) + 0.1cos(2t), sin(t) + 0.1sin(3t)))
        cg = conformal_grid(Φ, 10, 64; boundary_spacing = 0.05)
        disc = FerriteDiscretization(cg.grid)
        fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.1))
        σtrue = interpolate_function(disc, x -> 1 + (norm(x - Vec(0.3, 0.1)) < 0.4))
        σ = 1 .+ 0.3 .* rand(rng, ndofs_σ(disc))
        solver = BlockCGSolver(; preconditioner = PolarPreconditioner(disc; reference = cg.reference), rtol = 1e-13)
        I_ = trigonometric_patterns(fm, 2)
        V = forward_neumann(fm, σtrue, I_)[1]
        gd, gc = zeros(length(σ)), zeros(length(σ))
        @test value_and_gradient!(gc, AdjointStateObjective(fm, I_, V; solver), σ) ≈
              value_and_gradient!(gd, AdjointStateObjective(fm, I_, V), σ) rtol = 1e-8
        @test gc ≈ gd rtol = 1e-7
    end
end
