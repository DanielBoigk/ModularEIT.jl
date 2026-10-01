# Newest vertex bisection of triangle meshes: conformity, orientation, shape regularity, facet
# sets, levels, and the finite element layer on refined triangle meshes.
using ModularEIT
using ModularEITFerrite
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Random
using Test

# every interior edge is shared by exactly two triangles (no hanging nodes)
function conforming(grid)
    count = Dict{Tuple{Int, Int}, Int}()
    for c in getcells(grid), (a, b) in Ferrite.facets(c)
        k = minmax(a, b)
        count[k] = get(count, k, 0) + 1
    end
    # a hanging node would leave a long edge and its two halves each used once; detect it by
    # checking that no boundary edge (used once) has a node in its interior on another edge
    nodes_on_edges = Set{Int}()
    for (k, n) in count
        n == 1 || continue
        a, b = k
        pa, pb = get_node_coordinate(grid, a), get_node_coordinate(grid, b)
        for (k2, _) in count, v in k2
            v in k && continue
            pv = get_node_coordinate(grid, v)
            # v strictly inside segment ab?
            t = (pv - pa) ⋅ (pb - pa) / ((pb - pa) ⋅ (pb - pa))
            0 < t < 1 && norm(pa + t * (pb - pa) - pv) < 1e-12 && push!(nodes_on_edges, v)
        end
    end
    return all(<=(2), values(count)) && isempty(nodes_on_edges)
end

signed_area(grid, c) = (x = [get_node_coordinate(grid, n) for n in getcells(grid, c).nodes];
                        ((x[2] - x[1])[1] * (x[3] - x[1])[2] - (x[2] - x[1])[2] * (x[3] - x[1])[1]) / 2)

function min_angle(grid)
    m = π
    for c in getcells(grid)
        x = [get_node_coordinate(grid, n) for n in c.nodes]
        for i in 1:3
            a, b = x[mod1(i + 1, 3)] - x[i], x[mod1(i + 2, 3)] - x[i]
            m = min(m, acos(clamp(a ⋅ b / (norm(a) * norm(b)), -1, 1)))
        end
    end
    return m
end

@testset "newest vertex bisection" begin
    rng = MersenneTwister(8)
    base = generate_grid(Triangle, (4, 4))

    @testset "local refinement is conforming, oriented and area preserving" begin
        am = AdaptiveMesh(base; maxlevel = 30)
        @test getncells(current_grid(am)) == 32
        refine_mesh!(am, [5])
        g = current_grid(am)
        @test getncells(g) > 32
        @test g isa Grid{2, Triangle}
        @test conforming(g)
        @test all(c -> signed_area(g, c) > 0, 1:getncells(g))
        @test sum(c -> signed_area(g, c), 1:getncells(g)) ≈ 4.0
        @test !is_nonconforming(g)
        # many random refinements: still conforming, angles bounded below (finitely many
        # similarity classes)
        α0 = min_angle(base)
        for _ in 1:12
            n = getncells(current_grid(am))
            refine_mesh!(am, unique(rand(rng, 1:n, 4)))
        end
        g = current_grid(am)
        @test conforming(g)
        @test all(c -> signed_area(g, c) > 0, 1:getncells(g))
        @test sum(c -> signed_area(g, c), 1:getncells(g)) ≈ 4.0
        @test min_angle(g) >= α0 / 2 - 1e-12
    end

    @testset "uniform refinement, levels and the level cap" begin
        am = AdaptiveMesh(base; maxlevel = 4)
        refine_mesh!(am)                        # every triangle bisected once
        @test getncells(current_grid(am)) == 64
        refine_mesh!(am)
        @test getncells(current_grid(am)) == 128
        @test all(==(2), cell_levels(am))
        @test max_level(am) == 4
        g = current_grid(am)
        @test all(c -> signed_area(g, c) ≈ 4 / 128, 1:getncells(g))
        for _ in 1:4
            refine_mesh!(am, [1])
        end
        @test maximum(cell_levels(am)) <= 4      # cells at the cap are not refined
        # coarsening everything (repeatedly) returns to the initial mesh
        for _ in 1:10
            coarsen_mesh!(am, collect(1:getncells(current_grid(am))))
        end
        @test getncells(current_grid(am)) == 32
        @test all(==(0), cell_levels(am))
    end

    @testset "facet sets follow the refinement" begin
        am = AdaptiveMesh(base)
        for _ in 1:5
            refine_mesh!(am, unique(rand(rng, 1:getncells(current_grid(am)), 6)))
        end
        g = current_grid(am)
        disc = FerriteDiscretization(g)
        len(set) = sum(f -> ModularEITFerrite._facet_measure(disc, f), set)
        for name in ("left", "right", "top", "bottom")
            @test len(getfacetset(g, name)) ≈ 2.0
        end
        @test len(disc.boundary_facets) ≈ 8.0
        @test Set(disc.boundary_facets) == union(Set.(getfacetset(g, n) for n in ("left", "right", "top", "bottom"))...)

        circle = redirect_stdout(devnull) do
            togrid(joinpath(@__DIR__, "..", "data", "circle.msh"))
        end
        amc = AdaptiveMesh(circle)
        refine_mesh!(amc, collect(1:50))
        dc = FerriteDiscretization(current_grid(amc))
        @test sum(f -> ModularEITFerrite._facet_measure(dc, f), getfacetset(current_grid(amc), "boundary")) ≈
              sum(f -> ModularEITFerrite._facet_measure(FerriteDiscretization(circle), f), getfacetset(circle, "boundary"))
        @test haskey(current_grid(amc).cellsets, "domain")
        @test length(getcellset(current_grid(amc), "domain")) == getncells(current_grid(amc))
    end

    @testset "finite elements on refined triangle meshes" begin
        am = AdaptiveMesh(generate_grid(Triangle, (8, 8)))
        refine_mesh!(am, [3, 17, 30])
        refine_mesh!(am, [1, 2])
        g = current_grid(am)
        # conforming: continuous σ spaces work as well
        disc = FerriteDiscretization(g; ip_σ = Lagrange{RefTriangle, 1}())
        mats = FEMatrices(disc)
        x1 = interpolate_function(disc, x -> x[1]; field = :u)
        @test x1' * mats.K_u * x1 ≈ 4.0
        els = angular_electrodes(disc, 8)
        fm = ForwardModel(disc, CompleteElectrodeModel(els, 0.1))
        σtrue = 1 .+ rand(rng, ndofs_σ(disc))
        I = trigonometric_patterns(fm, 2)
        U, _ = forward_neumann(fm, σtrue, I)
        obj = AdjointStateObjective(fm, I, U)
        σ = 1 .+ rand(rng, ndofs_σ(disc))
        gr = zeros(length(σ))
        value_and_gradient!(gr, obj, σ)
        δ = randn(rng, length(σ))
        fd = (objective_value(obj, σ .+ 1e-5 .* δ) - objective_value(obj, σ .- 1e-5 .* δ)) / 2e-5
        @test dot(gr, δ) ≈ fd rtol = 1e-6

        # piecewise constants are inherited exactly
        d0 = FerriteDiscretization(g)
        σ0 = rand(rng, ndofs_σ(d0))
        refine_mesh!(am, [4, 9])
        d1 = FerriteDiscretization(current_grid(am))
        σ1 = transfer_conductivity(d0, σ0, d1)
        @test sum(FEMatrices(d1).M_σ * σ1) ≈ sum(FEMatrices(d0).M_σ * σ0)
        @test Set(round.(σ1; digits = 12)) ⊆ Set(round.(σ0; digits = 12))
    end

    @testset "other cell types warn" begin
        tet = generate_grid(Tetrahedron, (1, 1, 1))
        am = @test_logs (:warn, r"not implemented") AdaptiveMesh(tet)
        @test current_grid(am) === tet
    end
end
