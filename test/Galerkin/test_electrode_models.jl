# Electrode models turned into discrete forward models: injection P, measurement Q, the CEM
# augmentation, Neumann (current-driven) and Dirichlet (voltage-driven) forward solves.
using ModularEIT
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Random
using Test

# random current patterns that sum to zero (charge conservation)
function zero_sum_patterns(rng, m, s)
    X = randn(rng, m, s)
    return X .- sum(X; dims = 1) ./ m
end

@testset "electrode models" begin
    rng = MersenneTwister(5)
    grid = redirect_stdout(devnull) do
        togrid(joinpath(@__DIR__, "..", "data", "circle.msh"))          # unit disc, P1 triangles
    end
    disc = FerriteDiscretization(grid)
    nσ = ndofs_σ(disc)
    σ1 = ones(nσ)
    σ = 1 .+ rand(rng, nσ)
    L = 16

    @testset "angular electrodes" begin
        els = angular_electrodes(disc, L; coverage = 0.5)
        @test length(els) == L
        @test all(!isempty, els)
        @test isempty(intersect(Set.(els)...))
        lens = [electrode_length(disc, e) for e in els]
        @test sum(lens) ≈ π rtol = 0.03                      # half of the perimeter 2π
        @test all(l -> isapprox(l, π / L; rtol = 0.25), lens)
    end
    els = angular_electrodes(disc, L; coverage = 0.5)

    @testset "continuum model" begin
        fm = ForwardModel(disc, ContinuumModel())
        @test fm isa AbstractForwardModel
        nb = length(disc.boundary_dofs)
        @test n_inject(fm) == n_measure(fm) == n_control(fm) == nb
        @test size(fm.P) == (ndofs_u(disc), nb)
        @test sum(fm.P) ≈ 2π rtol = 1e-3          # P g = boundary load of the density g
        # homogeneous unit disc: R cos θ = cos θ, R cos 2θ = cos(2θ)/2
        G = trigonometric_patterns(fm, 2)          # cos θ, sin θ, cos 2θ, sin 2θ at boundary dofs
        V, X = forward_neumann(fm, σ1, G)
        @test size(V) == size(G) && size(X, 1) == fm.n
        @test norm(V[:, 1] - G[:, 1]) / norm(G[:, 1]) < 0.02
        @test norm(V[:, 3] - G[:, 3] / 2) / norm(G[:, 3] / 2) < 0.02
        # grounding (default): zero boundary mean, ∫_Γ u ds = 0
        w = FEMatrices(disc).M_Γ * ones(ndofs_u(disc))
        @test maximum(abs, w' * X) < 1e-10
        @test fm.measure_weights ≈ w[disc.boundary_dofs]
        # the old convention on request: boundary nodal values sum to zero
        fm_nodal = ForwardModel(disc, ContinuumModel(); grounding = :nodal)
        _, Xn = forward_neumann(fm_nodal, σ1, G)
        @test maximum(abs, sum(Xn[disc.boundary_dofs, :]; dims = 1)) < 1e-10
        @test_throws ArgumentError ForwardModel(disc, ContinuumModel(); grounding = :foo)
        # Dirichlet (voltage-driven) solve inverts the Neumann solve (DtN ∘ NtD = I on the data)
        V2, _ = forward_neumann(fm, σ, G)
        G2, _ = forward_dirichlet(fm, σ, V2)
        @test G2 ≈ G rtol = 1e-8
    end

    @testset "point electrodes" begin
        θ = [2π * (l - 1) / L for l in 1:L]
        pts = [Vec(cos(t), sin(t)) for t in θ]
        fm = ForwardModel(disc, PointElectrodeModel(pts))
        @test n_inject(fm) == n_measure(fm) == L
        @test all(==(1), sum(fm.P; dims = 1))
        @test count(!iszero, fm.P) == L
        I1, I2 = eachcol(zero_sum_patterns(rng, L, 2))
        V1, _ = forward_neumann(fm, σ, I1)
        V2, _ = forward_neumann(fm, σ, I2)
        @test dot(I2, V1) ≈ dot(I1, V2)            # reciprocity
        # voltage-driven on the point nodes (zero current elsewhere) inverts the current-driven solve
        I3, _ = forward_dirichlet(fm, σ, V1)
        @test I3 ≈ I1 rtol = 1e-8
    end

    @testset "gap model" begin
        fm = ForwardModel(disc, GapModel(els))
        @test n_inject(fm) == n_measure(fm) == L
        @test vec(sum(fm.P; dims = 1)) ≈ ones(L)   # ∫_{e_ℓ} φᵢ / |e_ℓ| sums to one: charge conserved
        @test fm.Q ≈ sparse(fm.P')                 # measurement = mean potential on the electrode
        I = zero_sum_patterns(rng, L, 2)
        V, _ = forward_neumann(fm, σ, I)
        @test dot(I[:, 2], V[:, 1]) ≈ dot(I[:, 1], V[:, 2])
        @test dot(I[:, 1], V[:, 1]) > 0            # dissipated power
        wΓ = FEMatrices(disc).M_Γ * ones(ndofs_u(disc))
        @test maximum(abs, wΓ' * forward_neumann(fm, σ, I)[2]) < 1e-10   # ∫_Γ u ds = 0

        # separate injection and measurement electrodes: inject on odd, measure on even electrodes
        inj, meas = els[1:2:end], els[2:2:end]
        fm_sep = ForwardModel(disc, GapModel(inj; measure = meas))
        @test n_inject(fm_sep) == L ÷ 2 && n_measure(fm_sep) == L ÷ 2
        Isep = zero_sum_patterns(rng, L ÷ 2, 3)
        Ifull = zeros(L, 3)
        Ifull[1:2:end, :] .= Isep
        Vsep, _ = forward_neumann(fm_sep, σ, Isep)
        Vfull, _ = forward_neumann(fm, σ, Ifull)
        # voltages agree up to the (arbitrary) ground
        Δ = Vsep - Vfull[2:2:end, :]
        @test maximum(abs, Δ .- sum(Δ; dims = 1) ./ size(Δ, 1)) < 1e-10
    end

    @testset "complete electrode model" begin
        z = 0.05
        fm = ForwardModel(disc, CompleteElectrodeModel(els, z))
        nu = ndofs_u(disc)
        @test fm.n == nu + L
        A = system_matrix!(fm, σ)
        @test issymmetric(A)
        @test norm(A * ones(fm.n)) < 1e-10         # null space: constants on (u, U) jointly
        @test fm.Q == sparse(1:L, nu .+ (1:L), ones(L), L, fm.n)
        I = zero_sum_patterns(rng, L, 2)
        U, X = forward_neumann(fm, σ, I)
        @test maximum(abs, sum(U; dims = 1)) < 1e-10   # grounding Σ U_ℓ = 0
        @test U == X[nu+1:end, :]
        @test dot(I[:, 2], U[:, 1]) ≈ dot(I[:, 1], U[:, 2])
        @test dot(I[:, 1], U[:, 1]) > 0
        # voltage-driven CEM inverts the current-driven one
        I2, _ = forward_dirichlet(fm, σ, U)
        @test I2 ≈ I rtol = 1e-8

        # large contact impedance: the current density under an electrode becomes uniform, so
        # U_ℓ ≈ (gap-model voltage) + z I_ℓ / |e_ℓ|
        zbig = 50.0
        fm_big = ForwardModel(disc, CompleteElectrodeModel(els, zbig))
        fm_gap = ForwardModel(disc, GapModel(els))
        Ubig, _ = forward_neumann(fm_big, σ, I)
        Ugap, _ = forward_neumann(fm_gap, σ, I)
        lens = [electrode_length(disc, e) for e in els]
        pred = Ugap .+ zbig .* I ./ lens
        pred .-= sum(pred; dims = 1) ./ L
        @test norm(Ubig - pred) / norm(Ubig) < 1e-3

        # measuring on a subset of electrodes (not on the current-carrying ones)
        fm_sub = ForwardModel(disc, CompleteElectrodeModel(els, z; measure = 2:2:L))
        @test n_measure(fm_sub) == L ÷ 2
        Usub, _ = forward_neumann(fm_sub, σ, I)
        @test Usub ≈ U[2:2:L, :]
        # contact impedances per electrode
        fm_vec = ForwardModel(disc, CompleteElectrodeModel(els, fill(z, L)))
        @test forward_neumann(fm_vec, σ, I)[1] ≈ U
    end

    @testset "block CG gives the direct-solver answer" begin
        fm = ForwardModel(disc, CompleteElectrodeModel(els, 0.05))
        I = zero_sum_patterns(rng, L, 4)
        Ud, _ = forward_neumann(fm, σ, I; solver = DirectSolver())
        for pc in (:amg, :jacobi)
            Uc, _ = forward_neumann(fm, σ, I; solver = BlockCGSolver(; preconditioner = pc, rtol = 1e-12))
            @test Uc ≈ Ud rtol = 1e-7
        end
    end
end
