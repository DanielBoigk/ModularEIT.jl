using LinearAlgebra, SparseArrays, Ferrite


@testset "SVD" begin
    
    conductivity = (x) -> 1.1 + sin(x[1]) * cos(x[2])
    n = 63 
    grid = generate_grid(Quadrilateral, (n, n));
    ∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
    fe  = FerriteFESpace{RefQuadrilateral}(grid,2,3,∂Ω)
    cond_vec = project_function_to_fem(fe, conductivity)
    cond_vec .= min.(max.(cond_vec,1e-6),1.0)
    G_full = real_fourier_basis(8)
    rhs_dict = Dict()
    Threads.@threads for i in 2:256
        M = make_boundary(G_full[:, i],64)
        itp = interpolate_array_2D(M)
        rhs_dict[i] = assemble_rhs_func(fe, itp)
    end
    K = assemble_L(fe, cond_vec)
    K_fac = cholesky(K)
    mode_dict = Vector{Any}(undef, 255)
    @time begin
    Threads.@threads for i in 2:256
        mode_dict[i-1] = create_mode_from_g(fe, rhs_dict[i], K_fac, normalize =  true)
    end
    mode_svd_noise, _ = svd_on_modes(mode_dict,fe)


    fn1 = mode_svd_noise[3].f
    gn1 = mode_svd_noise[3].g
    fn2 = mode_svd_noise[49].f
    gn2 = mode_svd_noise[49].g

    # Basically I test multilinearity of SVD vectors
    diff = fe.down(K \ fe.up(0.6*gn2+0.4*gn1)) - (0.6*fn2+0.4*fn1)
    diff.-= mean(diff)
    @test norm(diff) < 1e-12
end
end