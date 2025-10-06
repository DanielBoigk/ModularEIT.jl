using Ferrite, ModularEIT, Test, TypedPolynomials, SparseArrays, LinearAlgebra

@testset "BilinearMap.jl" begin

    grid = generate_grid(Quadrilateral, (16, 16))
    ∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)

    order = 3
    qr_order = 5

    fe = FerriteFESpace{RefQuadrilateral}(grid, order, qr_order, ∂Ω)

    @polyvar x y
    p = 2x * y^2 + y - 2x
    q = 3x * y^3 - 4x + 4
    function nabladotnabla(a, b)
        diff_a = differentiate(a, (x, y))
        diff_b = differentiate(b, (x, y))
        # There was a small typo in your original function here (diff_b[2] twice)
        # Correcting it to diff_a[2]*diff_b[2]
        return diff_a[1] * diff_b[1] + diff_a[2] * diff_b[2]
    end
    r = nabladotnabla(p, q)

    function get_func(p)
        (z) -> p(x => z[1], y => z[2])
    end

    p_func = get_func(p)
    q_func = get_func(q)
    r_func = get_func(r)


    p_vec = ModularEIT.project_function_to_fem(fe, p_func)
    q_vec = ModularEIT.project_function_to_fem(fe, q_func)
    r_vec = ModularEIT.project_function_to_fem(fe, r_func)

    r_test = calculate_bilinear_map(fe, p_vec, q_vec)

    norm_of_vec = norm(r_test - r_vec)
    @test norm_of_vec < 1e-9

end
