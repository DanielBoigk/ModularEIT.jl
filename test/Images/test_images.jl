# Finite element functions ↔ pixel images (2D meshes).
using ModularEIT
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Random
using Test

@testset "images" begin
    rng = MersenneTwister(31)

    @testset "pixel-matched quadrilateral mesh: exact in both directions" begin
        m, n = 6, 4                                           # 6 columns (x), 4 rows (y)
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (m, n)))
        σ = rand(rng, ndofs_σ(disc))
        img = to_image(disc, σ, n, m)
        @test size(img) == (n, m)
        # row 1 is the top of the domain, column 1 the left: cell (col j, row-from-bottom r)
        for i in 1:n, j in 1:m
            c = (n - i) * m + j
            @test img[i, j] == σ[only(celldofs(disc.dh_σ, c))]
        end
        @test from_image(disc, img) ≈ σ
        @test from_image(disc, img; method = :l2) ≈ σ
        img2 = rand(rng, n, m)
        @test to_image(disc, from_image(disc, img2), n, m) ≈ img2
        # orientation
        y = interpolate_function(disc, x -> x[2])
        x = interpolate_function(disc, x -> x[1])
        @test to_image(disc, y, n, m)[1, 1] > to_image(disc, y, n, m)[n, 1]
        @test to_image(disc, x, n, m)[1, 1] < to_image(disc, x, n, m)[1, m]
    end

    @testset "linear functions are reproduced ($name)" for (name, grid, field, ipσ) in (
            ("P1 potential, triangles", generate_grid(Triangle, (7, 5)), :u, nothing),
            ("P1 conductivity, triangles", generate_grid(Triangle, (7, 5)), :σ, Lagrange{RefTriangle, 1}()),
            ("Q1 potential, hanging nodes", let am = AdaptiveMesh(generate_grid(Quadrilateral, (4, 4)); maxlevel = 4)
                refine_mesh!(am, [1, 6, 7]); current_grid(am)
            end, :u, nothing))
        disc = ipσ === nothing ? FerriteDiscretization(grid) : FerriteDiscretization(grid; ip_σ = ipσ)
        f(x) = 1 + 2x[1] - 3x[2]
        c = interpolate_function(disc, f; field)
        n, m = 20, 30
        img = to_image(disc, c, n, m; field)
        xs = [-1 + (j - 0.5) * 2 / m for j in 1:m]
        ys = [1 - (i - 0.5) * 2 / n for i in 1:n]
        @test img ≈ [f((x, y)) for y in ys, x in xs]
        @test from_image(disc, img; field) ≈ c                 # bilinear sampling is exact for linear f
    end

    @testset "disc mesh: outside pixels, round trips" begin
        disc = FerriteDiscretization(redirect_stdout(devnull) do
            togrid(joinpath(@__DIR__, "..", "data", "circle.msh"))
        end)
        f(x) = x[1]^2 + x[2]
        u = interpolate_function(disc, f; field = :u)
        img = to_image(disc, u, 128, 128; field = :u)
        @test count(isnan, img) / length(img) ≈ 1 - π / 4 rtol = 0.05
        @test all(isfinite, to_image(disc, u, 16, 16; field = :u, outside = 0.0))
        img256 = to_image(disc, u, 256, 256; field = :u)
        for method in (:interpolate, :l2)
            u2 = from_image(disc, img256; field = :u, method)
            @test norm(u2 - u) / norm(u) < 2e-2
        end
        σ = interpolate_function(disc, f)
        σ2 = from_image(disc, to_image(disc, σ, 256, 256); method = :l2)
        @test norm(σ2 - σ) / norm(σ) < 2e-2
    end

    @testset "L² method preserves the integral" begin
        disc = FerriteDiscretization(generate_grid(Triangle, (9, 7)))
        img = rand(rng, 40, 40)
        σ = from_image(disc, img; method = :l2)
        @test sum(FEMatrices(disc).M_σ * σ) ≈ sum(img) * 4 / length(img) rtol = 1e-2
    end

    @testset "ImageMap: precomputed, reusable" begin
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (5, 5)))
        im = ImageMap(disc, 12, 10; field = :u)
        @test size(im.S) == (12 * 10, ndofs_u(disc))
        u = randn(rng, ndofs_u(disc))
        @test to_image(im, u) ≈ to_image(disc, u, 12, 10; field = :u)
        @test from_image(im, to_image(im, u)) ≈ from_image(disc, to_image(im, u); field = :u)
        @test_throws DimensionMismatch from_image(im, rand(3, 3))
        @test_throws ArgumentError from_image(disc, rand(4, 4); method = :foo)
        @test_throws ArgumentError ImageMap(FerriteDiscretization(generate_grid(Hexahedron, (1, 1, 1))), 4, 4)
    end

    @testset "unit images: rectangle, [0, 1] scaling, round trip" begin
        rng = MersenneTwister(12)
        # pixel-aligned square mesh: the unit image and its inverse are exact
        disc = FerriteDiscretization(generate_grid(Quadrilateral, (16, 16)))
        σ = 0.5 .+ 2.5 .* rand(rng, ndofs_σ(disc))
        ui = unit_image(disc, σ, 16, 16)
        @test ui isa UnitImage
        @test size(ui.image) == (16, 16) && all(ui.mask)
        @test ui.bbox == (-1.0, 1.0, -1.0, 1.0)
        @test ui.range == extrema(σ)
        @test extrema(ui.image) == (0.0, 1.0)
        @test from_unit_image(disc, ui) ≈ σ
        # logarithmic scaling
        uil = unit_image(disc, σ, 16, 16; scale = :log)
        @test extrema(uil.image) == (0.0, 1.0)
        @test uil.image ≈ (log.(to_image(disc, σ, 16, 16)) .- log(minimum(σ))) ./ log(maximum(σ) / minimum(σ))
        @test from_unit_image(disc, uil) ≈ σ
        # fixed range (consistent scaling across a data set): values outside are clamped
        uif = unit_image(disc, σ, 16, 16; range = (1.0, 2.0))
        @test uif.range == (1.0, 2.0)
        @test all(0 .<= uif.image .<= 1)
        @test uif.image ≈ clamp.((to_image(disc, σ, 16, 16) .- 1.0) ./ 1.0, 0, 1)

        # rectangular domain, square array: the rectangle is widened (centred) to square pixels
        rect = FerriteDiscretization(generate_grid(Quadrilateral, (12, 4), Vec(0.0, 0.0), Vec(3.0, 1.0)))
        σr = 1 .+ rand(rng, ndofs_σ(rect))
        ur = unit_image(rect, σr, 30, 30; outside = -1.0)
        @test ur.bbox == (0.0, 3.0, -1.0, 2.0)
        @test count(ur.mask) == 30 * 10                       # the middle 10 rows cover the domain
        @test all(ur.image[.!ur.mask] .== -1.0)
        @test all(0 .<= ur.image[ur.mask] .<= 1)
        # ... or stretched to the array (non-square pixels, no outside pixels)
        us = unit_image(rect, σr, 30, 30; pixels = :stretch)
        @test us.bbox == (0.0, 3.0, 0.0, 1.0) && all(us.mask)
        # pixel-aligned rectangle with square pixels: exact round trip
        ua = unit_image(rect, σr, 4, 12)
        @test ua.bbox == (0.0, 3.0, 0.0, 1.0)
        @test from_unit_image(rect, ua) ≈ σr

        # a disk: square box around it, zero outside
        disk = FerriteDiscretization(polar_grid(6, 48))
        ud = unit_image(disk, fill(2.0, ndofs_σ(disk)), 40, 40)
        @test ud.bbox[2] - ud.bbox[1] ≈ ud.bbox[4] - ud.bbox[3]
        @test all(ud.image[.!ud.mask] .== 0)
        @test 0.7 < count(ud.mask) / 1600 < π / 4 + 0.02           # ≈ disk / square
        @test all(ud.image[ud.mask] .== 0)                          # constant σ maps to 0

        @test_throws ArgumentError unit_image(disc, σ, 16, 16; scale = :cubic)
        @test_throws ArgumentError unit_image(disc, σ, 16, 16; pixels = :round)
        @test_throws ArgumentError unit_image(disc, σ .- 1, 16, 16; scale = :log)
        @test_throws ArgumentError unit_image(disc, σ, 16, 16; range = (2.0, 1.0))
    end
end
