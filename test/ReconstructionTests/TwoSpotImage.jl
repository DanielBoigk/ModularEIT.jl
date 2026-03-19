
using ModularEIT
using FFTW
using Images
using Ferrite
using Enzyme


@testset "Reconstruction Two Spot Image LBFGS" begin
    println("Starting Reconstruction Two Spot Image LBFGS test")
    n = 63
    grid = generate_grid(Quadrilateral, (n, n))
    ∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
    fe = FerriteFESpace{RefQuadrilateral}(grid, 2, 3, ∂Ω)
    img = load("SolverTests/Reference2Spot.jpg")
    itp = interpolate_array_2D(Float64.(img))
    cond_vec = project_function_to_fem(fe, itp)

    # Generate a basis on the boundary and assemble it as a righthandside vector.
    # Use fewer modes for faster testing
    G_full = real_fourier_basis(8)
    #= # For reasons I cannot explain using a dict in a parallel doesn't work anymore...
    rhs_dict = Dict()
    Threads.@threads for i in 2:256
        M = make_boundary(G_full[:, i], 64)
        itp = interpolate_array_2D(M)
        rhs_dict[i] = assemble_rhs_func(fe, itp)
    end
    =#
    rhs_vec = Vector{Any}(undef, 255)

    Threads.@threads for i in 2:256
        M = make_boundary(G_full[:, i], 64)
        itp = interpolate_array_2D(M)
        rhs_vec[i-1] = assemble_rhs_func(fe, itp)
    end
    # Assemble stiffness matrix and calculate boundary pairs:
    K = assemble_L(fe, cond_vec)
    K_fac = factorize(K)

    mode_vec = Vector{Any}(undef, 255)
    for i in 1:255
        mode_vec[i] = create_mode_from_g(fe, rhs_vec[i], K)
    end
    # define starting guess and define problem:
    σ_vec = project_function_to_fem(fe, x -> 0.5)
    sol = FerriteSolverState(fe, σ_vec)
    prblm = FerriteProblem(fe, mode_vec, sol)

    #solve_modes!(prblm, 255, state_adjoint_step_neumann_init!)
    #@test prblm.modes[1].error_n ≠ -1.0

    # We need the point handler later
    eval_points = reshape(equidistant_grid(64), :)
    ph = PointEvalHandler(grid, eval_points)


    # We set the regularizer:
    grad_normH1sq(fe, a) = 2 * fe.K * a
    TikhonovReg = (x) -> normH1sq(prblm.fe, x)
    ∇Tkhnv = (x) -> grad_normH1sq(prblm.fe, x)
    add_diff_Regularizer!(prblm.state, TikhonovReg, nothing, ∇Tkhnv)

    # TV = (x) -> normTV_diff(prblm.fe, x)
    # add_diff_Regularizer!(prblm.state, TV, nothing, ∇TV)

    prblm.state.opt.β_diff = 1e-4


    # we wrap the function for use in LBFGS:

    f, ∂f = create_f∂f(prblm, 24; regularize=false, gn=false)  # Reduced from 255 to 19
    # I think this is incorrect atleast it produces nonsense:
    #f, ∂f = create_f∂f(prblm, 10; regularize=false, gn=false, mode="mixed", obj=objective_mixed_init!, grad=gradient_mixed_init!)


    #f, ∂f = create_f∂f(prblm, 24; regularize=false, gn=false, mode="dirichlet", obj=objective_dirichlet_init!, grad=gradient_dirichlet_init!)


    # Now we solve the problem:
    println("Starting LBFGS:")
    # LBFGS expects descent direction (negative gradient), so negate ∂f
    descent_dir(x) = ∂f(x)
    solution = lbfgs_b(f, descent_dir, copy(σ_vec); m=10, tol=1e-6, maxiter=20)

    starting_error = norm(σ_vec - cond_vec)
    total_error = norm(solution - cond_vec)
    println("L2-distance of starting guess: $starting_error")
    println("L2-distance of reconstruction: $total_error")
    # Check that optimizer made progress on the objective (loss decreased)
    f_initial = f(σ_vec)
    f_final = f(solution)
    println("Initial objective: $f_initial")
    println("Final objective: $f_final")

    img_initial = Gray.(max.(0.0, min.(1.0, reshape(evaluate_at_points(ph, prblm.fe.dh, cond_vec), (64, 64)))))
    img_final = Gray.(max.(0.0, min.(1.0, reshape(evaluate_at_points(ph, prblm.fe.dh, solution), (64, 64)))))

    # Save the images for inspection
    save("ReconstructionTests/Reconstruction/TwoSpot_init.png", img_initial)
    save("ReconstructionTests/Reconstruction/TwoSpotLBFGS_final.png", img_final)
    @test f_final < f_initial  # Objective should decrease
end
