# On identical meshes the Gridap back end reproduces the Ferrite back end.
using ModularEIT
using ModularEITFerrite
using ModularEITGridap
using Ferrite: Ferrite, generate_grid, Triangle, Quadrilateral, Lagrange, RefTriangle
using LinearAlgebra
using SparseArrays
using Random
using Test

isdefined(Main, :gridap_model) || include("helpers.jl")

# u dofs of the Gridap discretization in the order of the Ferrite dofs
function u_permutation(df, dg)
    xf = interpolate_function(df, p -> p[1]; field = :u)
    yf = interpolate_function(df, p -> p[2]; field = :u)
    xg = interpolate_function(dg, p -> p[1]; field = :u)
    yg = interpolate_function(dg, p -> p[2]; field = :u)
    return dof_permutation(xg, yg, xf, yf)
end

# electrodes of the Ferrite discretization as facet ids of the Gridap one (same facet midpoints)
function gridap_electrodes(df, dg, electrodes)
    key(m) = (round(m[1]; digits = 9), round(m[2]; digits = 9))
    ids = Dict(key(ModularEIT._facet_midpoint(dg, f)) => f for f in ModularEIT._boundary_facets(dg))
    return [[ids[key(ModularEIT._facet_midpoint(df, f))] for f in e] for e in electrodes]
end

@testset "Gridap back end reproduces Ferrite" begin
    rng = MersenneTwister(3)
    for (name, grid) in (("P1/P0 triangles", generate_grid(Triangle, (6, 6))),
                         ("Q1/Q0 quadrilaterals", generate_grid(Quadrilateral, (5, 5))))
        @testset "$name" begin
            df = FerriteDiscretization(grid)
            dg = GridapDiscretization(gridap_model(grid))
            @test ndofs_u(dg) == ndofs_u(df)
            @test ndofs_σ(dg) == ndofs_σ(df)
            p = u_permutation(df, dg)
            # σ dofs are numbered by cells in both back ends
            cx(d) = interpolate_function(d, x -> x[1] + 10x[2])
            @test cx(dg) ≈ cx(df)

            mf, mg = FEMatrices(df), FEMatrices(dg)
            for f in (:M_u, :K_u, :M_Γ)
                @test getfield(mg, f)[p, p] ≈ getfield(mf, f) atol = 1e-12
            end
            @test mg.M_σ ≈ mf.M_σ
            @test sort(ModularEIT._boundary_dofs(dg)) == sort(p[ModularEIT._boundary_dofs(df)])

            σ = 1 .+ rand(rng, ndofs_σ(df))
            ct = ConductivityTensor(dg)
            L = assemble_weighted_stiffness!(copy(ct.pattern), ct, σ)
            @test L ≈ assemble_weighted_stiffness(dg, σ)
            @test L[p, p] ≈ assemble_weighted_stiffness(df, σ)

            # forward problems of all electrode models: same voltages
            ef = angular_electrodes(df, 8; coverage = 0.6)
            eg = angular_electrodes(dg, 8; coverage = 0.6)
            @test all(!isempty, ef)
            @test Set.(eg) == Set.(gridap_electrodes(df, dg, ef))
            @test [electrode_length(dg, e) for e in eg] ≈ [electrode_length(df, e) for e in ef]
            for (mf_, mg_) in ((CompleteElectrodeModel(ef, 0.1), CompleteElectrodeModel(eg, 0.1)),
                               (GapModel(ef), GapModel(eg)),
                               (PointElectrodeModel([(cos(t), sin(t)) for t in 0.1:0.7:6]),   # (no ties between nodes)
                                PointElectrodeModel([(cos(t), sin(t)) for t in 0.1:0.7:6])))
                fmf, fmg = ForwardModel(df, mf_), ForwardModel(dg, mg_)
                I = trigonometric_patterns(fmf, 3)
                @test trigonometric_patterns(fmg, 3) ≈ I
                Vf, _ = forward_neumann(fmf, σ, I)
                Vg, _ = forward_neumann(fmg, σ, I)
                @test Vg ≈ Vf rtol = 1e-10
            end
            # continuum model: measured values at the boundary dofs, in their own order
            fmf, fmg = ForwardModel(df, ContinuumModel()), ForwardModel(dg, ContinuumModel())
            bf, bg = ModularEIT._boundary_dofs(df), ModularEIT._boundary_dofs(dg)
            q = [findfirst(==(p[i]), bg) for i in bf]                  # Gridap position of each Ferrite boundary dof
            G = randn(rng, length(bf), 2)
            Gf = G .- (fmf.measure_weights' * G) ./ sum(fmf.measure_weights)
            Gg = zeros(length(bg), 2)
            Gg[q, :] = Gf
            Vf, _ = forward_neumann(fmf, σ, Gf)
            Vg, _ = forward_neumann(fmg, σ, Gg)
            @test Vg[q, :] ≈ Vf rtol = 1e-9

            # objective and gradient
            fmf, fmg = ForwardModel(df, CompleteElectrodeModel(ef, 0.1)), ForwardModel(dg, CompleteElectrodeModel(eg, 0.1))
            I = trigonometric_patterns(fmf, 3)
            data, _ = forward_neumann(fmf, 1 .+ rand(rng, ndofs_σ(df)), I)
            of, og = AdjointStateObjective(fmf, I, data), AdjointStateObjective(fmg, I, data)
            gf, gg = zeros(length(σ)), zeros(length(σ))
            @test value_and_gradient!(gg, og, σ) ≈ value_and_gradient!(gf, of, σ)
            @test gg ≈ gf

            # regularizers
            @test total_variation(dg, σ; ε = 0.1) ≈ total_variation(df, σ; ε = 0.1)
            for reg in ((d -> TotalVariationRegularizer(d; ε = 0.05)), (d -> TikhonovRegularizer(d; kind = :jump)))
                rf, rg = reg(df), reg(dg)
                @test objective_value(rg, σ) ≈ objective_value(rf, σ)
                @test gauss_newton_hessian(rg, σ) ≈ gauss_newton_hessian(rf, σ)
            end
            v = σ .+ 0.3 .* randn(rng, length(σ))
            zf = prox(TotalVariationRegularizer(df; ε = 0.0), v, 5.0; weights = lumped_mass(df))
            zg = prox(TotalVariationRegularizer(dg; ε = 0.0), v, 5.0; weights = lumped_mass(dg))
            @test zg ≈ zf rtol = 1e-6
        end
    end

    @testset "P2 potential, continuous P1 conductivity" begin
        grid = generate_grid(Triangle, (5, 5))
        df = FerriteDiscretization(grid; ip_u = Lagrange{RefTriangle, 2}(), ip_σ = Lagrange{RefTriangle, 1}())
        dg = GridapDiscretization(gridap_model(grid); order_u = 2, order_σ = 1)
        @test ndofs_u(dg) == ndofs_u(df)
        @test ndofs_σ(dg) == ndofs_σ(df)
        f = x -> 1.5 + 0.4x[1] - 0.3x[2]                               # in both σ spaces exactly
        σf, σg = interpolate_function(df, f), interpolate_function(dg, f)
        ct = ConductivityTensor(dg)
        @test assemble_weighted_stiffness!(copy(ct.pattern), ct, σg) ≈ assemble_weighted_stiffness(dg, σg)
        ef, eg = angular_electrodes(df, 6), angular_electrodes(dg, 6)
        fmf, fmg = ForwardModel(df, CompleteElectrodeModel(ef, 0.1)), ForwardModel(dg, CompleteElectrodeModel(eg, 0.1))
        I = trigonometric_patterns(fmf, 2)
        @test forward_neumann(fmg, σg, I)[1] ≈ forward_neumann(fmf, σf, I)[1] rtol = 1e-10
        h = x -> 1 + 0.5x[1]^2                                           # not in the σ space
        @test total_variation(dg, interpolate_function(dg, h); ε = 0.01) ≈
              total_variation(df, interpolate_function(df, h); ε = 0.01) rtol = 1e-10
        # gradient of a continuous-σ objective by finite differences
        data, _ = forward_neumann(fmg, σg .* 1.1, I)
        og = AdjointStateObjective(fmg, I, data)
        g = zeros(length(σg))
        value_and_gradient!(g, og, σg)
        δ = randn(MersenneTwister(5), length(σg))
        fd = (objective_value(og, σg .+ 1e-6δ) - objective_value(og, σg .- 1e-6δ)) / 2e-6
        @test dot(g, δ) ≈ fd rtol = 1e-5
    end
end
