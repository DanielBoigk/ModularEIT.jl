# Gradient test for ModularEIT adjoint gradient computation
# Compares adjoint gradient with finite difference approximation

using ModularEIT
using Ferrite
using LinearAlgebra
using Random
using Test

function run_gradient_test(; mesh_size=4, order=1, qr_order=2, seed=123)
    Random.seed!(seed)

    println("="^60)
    println("Running gradient test")
    println("Mesh size: $mesh_size×$mesh_size")
    println("FE order: $order")
    println("="^60)

    # 1. Create grid and finite element space
    grid = generate_grid(Quadrilateral, (mesh_size, mesh_size))
    ∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
    fe = FerriteFESpace{RefQuadrilateral}(grid, order, qr_order, ∂Ω)

    println("FE space: n=$(fe.n) dofs, m=$(fe.m) boundary dofs")

    # 2. Create true conductivity (constant + small random perturbation)
    σ_true = ones(fe.n) .+ 0.1 * randn(fe.n)
    σ_true = max.(σ_true, 0.1)  # ensure positivity

    # 3. Create random boundary current vector g (size m)
    g = randn(fe.m)
    # Ensure zero mean (current injection must sum to zero)
    g .-= mean(g)

    # 4. Compute corresponding boundary potential f using true conductivity
    K_true = assemble_L(fe, σ_true)  # stiffness matrix for true σ
    G = fe.up(g)  # expand to full vector
    u_true = K_true \ G  # solve forward problem
    f = fe.down(u_true)  # restrict to boundary
    f .-= mean(f)  # normalize potential

    println("Created boundary pair: ‖g‖=$(norm(g)), ‖f‖=$(norm(f))")

    # 5. Create mode using true conductivity
    mode = create_mode_from_fg(fe, f, g)

    # 6. Create solver state with true conductivity
    state = FerriteSolverState(fe, σ_true)

    # 7. Compute gradient using adjoint method
    gradient_neumann_init!(mode, state, fe)  # uses factorized solve
    grad_adjoint = copy(mode.δσ)

    println("Adjoint gradient computed: ‖∇J_adj‖=$(norm(grad_adjoint))")

    # 8. Finite difference gradient approximation
    ε = 1e-6
    grad_fd = zeros(fe.n)

    # Cache original state and mode
    σ_orig = copy(state.σ)
    mode_orig = deepcopy(mode)  # not strictly needed but safe

    # Function to compute objective J(σ) for given conductivity
    function compute_objective(σ::AbstractVector)
        # Create temporary solver state with this σ
        state_temp = FerriteSolverState(fe, σ)
        # Use the same mode (boundary data)
        mode_temp = deepcopy(mode_orig)
        # Compute objective (Neumann data misfit)
        objective_neumann_init!(mode_temp, state_temp, fe)
        return mode_temp.error_n
    end

    J0 = compute_objective(σ_orig)

    # Finite differences for each degree of freedom
    for i in 1:fe.n
        σ_pert = copy(σ_orig)
        σ_pert[i] += ε
        J_pert = compute_objective(σ_pert)
        grad_fd[i] = (J_pert - J0) / ε
    end

    println("FD gradient computed: ‖∇J_fd‖=$(norm(grad_fd))")

    # 9. Compare gradients
    diff = grad_adjoint - grad_fd
    rel_err = norm(diff) / (norm(grad_adjoint) + norm(grad_fd) + 1e-12)

    println("\nComparison results:")
    println("  ‖∇J_adj - ∇J_fd‖ = $(norm(diff))")
    println("  Relative error    = $(rel_err)")
    println("  max |diff|       = $(maximum(abs.(diff)))")
    println("  min |diff|       = $(minimum(abs.(diff)))")

    # 10. Test passes if relative error is small
    @test rel_err < 1e-4
    println("\n✅ Gradient test passed!")

    return grad_adjoint, grad_fd, rel_err
end

# Run the test when this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_gradient_test()
end
