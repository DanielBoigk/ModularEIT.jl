# Regrounding measured voltages and SVD of boundary data pairs under different inner products.
using ModularEIT
using ModularEITFerrite
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Random
using Test

@testset "pattern SVD" begin
    rng = MersenneTwister(17)
    circle = redirect_stdout(devnull) do
        togrid(joinpath(@__DIR__, "..", "data", "circle.msh"))
    end

    @testset "regrounding is exact and reversible" begin
        V = randn(rng, 20, 4)
        w1, w2 = rand(rng, 20), rand(rng, 20)
        A = reground(V, w1)
        @test maximum(abs, w1' * A) < 1e-12
        @test reground(A, w2) ≈ reground(V, w2)                 # shifting back and forth
        @test A .- A[1:1, :] ≈ V .- V[1:1, :]                   # differences unchanged
        @test reground(V, ones(20)) ≈ V .- sum(V; dims = 1) ./ 20

        disc = FerriteDiscretization(circle)
        fm = ForwardModel(disc, ContinuumModel())
        G = trigonometric_patterns(fm, 2)
        Vint, _ = forward_neumann(fm, ones(ndofs_σ(disc)), G)   # grounded with ∫_Γ u ds = 0
        Vnod = reground(fm, Vint, :nodal)
        @test maximum(abs, sum(Vnod; dims = 1)) < 1e-10
        @test reground(fm, Vnod, :integral) ≈ Vint atol = 1e-12
        # the same as solving with the nodal ground directly
        fmn = ForwardModel(disc, ContinuumModel(); grounding = :nodal)
        @test forward_neumann(fmn, ones(ndofs_σ(disc)), G)[1] ≈ Vnod atol = 1e-10
    end

    @testset "new pairs: orthonormal, sorted, valid data ($metric)" for metric in (:euclidean, :L2)
        disc = FerriteDiscretization(circle)
        fm = ForwardModel(disc, ContinuumModel())
        σ = 1 .+ rand(rng, ndofs_σ(disc))
        G = trigonometric_patterns(fm, 4)
        # mix the patterns so that the input is far from orthogonal
        G = G * (I + 0.5 * randn(rng, size(G, 2), size(G, 2)))
        V, _ = forward_neumann(fm, σ, G)
        p = pattern_svd(disc, fm, G, V; metric)
        @test size(p.currents) == size(G) && size(p.voltages) == size(V)
        @test issorted(p.values; rev = true) && all(>(0), p.values)
        @test p.currents' * p.Mi * p.currents ≈ I atol = 1e-10
        @test p.voltages' * p.Mv * p.voltages ≈ Diagonal(p.values .^ 2) atol = 1e-10
        # linear combinations of measured pairs are measured pairs of the same forward model
        Vnew, _ = forward_neumann(fm, σ, p.currents)
        w = metric === :L2 ? fm.measure_weights : ones(size(V, 1))
        @test reground(Vnew, w) ≈ p.voltages rtol = 1e-8
    end

    @testset "homogeneous unit disc: NtD singular values 1/k" begin
        disc = FerriteDiscretization(circle)
        fm = ForwardModel(disc, ContinuumModel())
        G = trigonometric_patterns(fm, 3)
        V, _ = forward_neumann(fm, ones(ndofs_σ(disc)), G)
        p = pattern_svd(disc, fm, G, V; metric = :L2)
        @test p.values ≈ [1, 1, 1 / 2, 1 / 2, 1 / 3, 1 / 3] rtol = 0.02
    end

    @testset "graded boundary mesh: the L² SVD is mesh independent, the Euclidean one is not" begin
        uniform = FerriteDiscretization(generate_grid(Quadrilateral, (32, 32)))
        am = AdaptiveMesh(generate_grid(Quadrilateral, (16, 16)); maxlevel = 4)
        bcells = [c for c in 1:256 if any(i -> i in (1, 16), fldmod1(c, 16))]
        right = filter(c -> fldmod1(c, 16)[2] == 16, bcells)          # refine the right side only
        refine_mesh!(am, right)
        refine_mesh!(am, [c for c in 1:getncells(current_grid(am)) if cell_levels(am)[c] == 1])
        graded = FerriteDiscretization(current_grid(am))
        vals(d, metric) = begin
            fm = ForwardModel(d, ContinuumModel())
            G = trigonometric_patterns(fm, 3)
            V, _ = forward_neumann(fm, ones(ndofs_σ(d)), G)
            pattern_svd(d, fm, G, V; metric).values
        end
        dL2 = norm(vals(graded, :L2) - vals(uniform, :L2)) / norm(vals(uniform, :L2))
        dE = norm(vals(graded, :euclidean) / vals(graded, :euclidean)[1] -
                  vals(uniform, :euclidean) / vals(uniform, :euclidean)[1])
        @test dL2 < 0.02
        @test dE > 5 * dL2
    end

    @testset "electrode models" begin
        disc = FerriteDiscretization(circle)
        els = angular_electrodes(disc, 16)
        fm = ForwardModel(disc, CompleteElectrodeModel(els, 0.05))
        Ic = trigonometric_patterns(fm, 4)
        U, _ = forward_neumann(fm, 1 .+ rand(rng, ndofs_σ(disc)), Ic)
        pe = pattern_svd(disc, fm, Ic, U; metric = :euclidean)
        pl = pattern_svd(disc, fm, Ic, U; metric = :L2)
        lens = [electrode_length(disc, e) for e in els]
        @test pl.Mv ≈ Diagonal(lens) && pl.Mi ≈ Diagonal(1 ./ lens)
        # nearly equal electrodes: both metrics give nearly the same spectrum up to the scale |e|
        @test pl.values ./ pe.values ≈ fill(sum(lens) / 16, 8) rtol = 0.02
        # explicit metrics
        pm = pattern_svd(disc, fm, Ic, U; metric = (Matrix(1.0I, 16, 16), Matrix(2.0I, 16, 16)))
        @test pm.values ≈ sqrt(2) .* pe.values
        fmp = ForwardModel(disc, PointElectrodeModel([Vec(cos(t), sin(t)) for t in 2π .* (0:7) ./ 8]))
        Ip = trigonometric_patterns(fmp, 2)
        Up, _ = forward_neumann(fmp, ones(ndofs_σ(disc)), Ip)
        @test_throws ArgumentError pattern_svd(disc, fmp, Ip, Up; metric = :L2)
        @test_throws ArgumentError pattern_svd(disc, fm, Ic, U; metric = :foo)
    end
end
