"""
Gradient test for the Neumann adjoint solver (gradient_neumann_init!).

This test verifies that the computed gradient ∂J/∂σ matches the finite-difference
approximation of the directional derivative.

The objective is J(σ) = d(b(σ), f) where:
- σ is the conductivity (the variable we're optimizing over)
- u(σ) is the solution to the forward PDE: ∇⋅(σ∇u) = 0 with boundary condition σ∂u/∂n = g
- b(σ) is the boundary trace of u(σ)
- f is the target boundary data
- d(·,·) is the data misfit (e.g., L2 norm squared)

The gradient should satisfy: ∇J(σ) ≈ [J(σ + εδσ) - J(σ)] / ε for small ε and any direction δσ.

More precisely, we check: lim_{ε→0} |J'(σ; δσ) - [J(σ + εδσ) - J(σ)] / ε| = 0
where J'(σ; δσ) = ⟨∇J(σ), δσ⟩ is the directional derivative.
"""

using ModularEIT
using Test
using LinearAlgebra
using Random
using Statistics

function get_node_coordinates(grid::Ferrite.Grid)
    """Extract x-coordinates of all nodes in the grid."""
    coords = grid.nodes
    x = zeros(length(coords))
    for (i, node) in enumerate(coords)
        x[i] = node.x[1]
    end
    return x
end

function test_neumann_gradient(; grid_size=15, n_modes=3, seed=42)
    """
    Test the gradient computation for the Neumann adjoint solver.

    Returns true if the gradient passes the finite-difference test, false otherwise.
    """
    println("\n" * "=" * 70)
    println("Neumann Adjoint Gradient Test")
    println("=" * 70)

    # Set random seed for reproducibility
    Random.seed!(seed)

    # Setup problem
    println("\n1. Setting up FEM space and modes...")
    grid = generate_grid(Quadrilateral, (grid_size, grid_size))
    ∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
    fe = FerriteFESpace{RefQuadrilateral}(grid, 2, 3, ∂Ω)

    # Create a reference conductivity
    x_coords = get_node_coordinates(grid)
    σ_ref = ones(fe.n) .+ 0.3 .* sin.(2π .* x_coords ./ maximum(x_coords))
    σ_ref = max.(σ_ref, 1e-6)  # Ensure positivity

    # Assemble reference stiffness matrix
    K_ref = assemble_L(fe, σ_ref)
    K_fac = factorize(K_ref)

    # Create modes with random boundary data
    mode_dict = Dict{Int64,FerriteEITMode}()
    for i in 1:n_modes
        # Random boundary source data
        g_vec = randn(fe.m)
        g_vec .-= mean(g_vec)
        G = fe.up(g_vec)

        # Random target boundary data
        f_vec = randn(fe.m)
        f_vec .-= mean(f_vec)

        # Create mode using reference conductivity
        mode_dict[i] = create_mode_from_g(fe, G, K_fac)
        # Override the target data
        mode_dict[i].f = copy(f_vec)
        mode_dict[i].F = fe.up(f_vec)
    end

    println("   - Grid: $(grid_size)×$(grid_size)")
    println("   - FE space dimension: $(fe.n)")
    println("   - Boundary DOFs: $(fe.m)")
    println("   - Number of modes: $(n_modes)")

    # Setup solver state with test conductivity
    println("\n2. Setting up solver state...")
    σ_test = 0.5 .+ 0.1 .* randn(fe.n)
    σ_test = max.(σ_test, 1e-6)

    sol = FerriteSolverState(fe, σ_test)
    prblm = FerriteProblem(fe, mode_dict, sol)

    # Compute objective and gradient at current point
    println("\n3. Computing objective and gradient at test point...")
    for (i, mode) in mode_dict
        objective_neumann_init!(mode, sol, fe)
        gradient_neumann_init!(mode, sol, fe)
    end

    J = sum(m.error_n for m in values(mode_dict))
    δJ = sum(m.δσ for m in values(mode_dict))

    println("   - Objective value: $J")
    println("   - Gradient norm: $(norm(δJ))")

    # Test gradient accuracy with finite differences
    println("\n4. Running finite-difference gradient test...")

    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    n_test_directions = 2

    all_rates = []

    for dir_idx in 1:n_test_directions
        println("\n   Testing direction $dir_idx of $n_test_directions...")

        # Random search direction normalized
        δσ = randn(fe.n)
        δσ .-= mean(δσ)  # Make it zero-mean
        δσ ./= norm(δσ)

        # Directional derivative from gradient
        J_grad_dir = dot(δJ, δσ)

        # Finite difference approximations for different step sizes
        errors = []

        for ε in epsilons
            σ_pert = σ_test .+ ε .* δσ
            σ_pert = max.(σ_pert, 1e-6)  # Enforce lower bound

            # Update state with perturbed conductivity
            sol.σ .= σ_pert
            update_L!(sol, fe, true)

            # Recompute objectives at perturbed point
            for (i, mode) in mode_dict
                objective_neumann_init!(mode, sol, fe)
            end
            J_pert = sum(m.error_n for m in values(mode_dict))

            # Finite difference approximation of directional derivative
            J_fd_dir = (J_pert - J) / ε

            # Error between gradient-based and FD-based approximations
            error = abs(J_grad_dir - J_fd_dir)
            push!(errors, error)
        end

        # Compute convergence rate
        if length(errors) >= 2
            log_eps = log.(epsilons)
            log_err = log.(errors)

            # Linear regression: log(error) = rate * log(ε) + intercept
            # Expected rate ≈ 1.0 for first-order accuracy
            A = [log_eps ones(length(log_eps))] \ log_err
            rate = A[1]
            push!(all_rates, rate)

            println("      Gradient norm: $(norm(δJ))")
            println("      Convergence rate: $(round(rate, digits=2)) (expect ≈ 1.0)")

            # Print a few sample errors for debugging
            for (ε, err) in zip(epsilons[1:3], errors[1:3])
                println("        ε = $(ε): error = $(err)")
            end
        end
    end

    # Summary
    println("\n" * "=" * 70)
    println("GRADIENT TEST SUMMARY")
    println("=" * 70)

    if !isempty(all_rates)
        avg_rate = mean(all_rates)
        println("Average convergence rate: $(round(avg_rate, digits=3))")
        println("Expected convergence rate: ≈ 1.0 (for first-order accurate gradient)")

        # Pass if convergence rate is between 0.8 and 1.2
        passed = avg_rate > 0.8 && avg_rate < 1.3

        if passed
            println("\n✓ PASS: Gradient is first-order accurate")
            return true
        else
            println("\n✗ FAIL: Convergence rate $(round(avg_rate, digits=3)) is outside expected range [0.8, 1.3]")
            println("This suggests the gradient computation may have an error.")
            return false
        end
    else
        println("ERROR: Could not compute convergence rate")
        return false
    end
end

@testset "NeumannAdjointGradientTest" begin
    println("\n\nStarting Neumann Adjoint Gradient Test")
    result = test_neumann_gradient(grid_size=15, n_modes=3, seed=42)
    @test result "Gradient test failed: computed gradient does not match finite differences"
end
