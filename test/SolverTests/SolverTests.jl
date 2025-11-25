using Test

using ModularEIT
using Ferrite
using IterativeSolvers
using Enzyme
using Interpolations
using Images

function real_fourier_basis(n::Int)
    N = 2^n
    t = collect(0:N-1) ./ N
    F = Matrix{Float64}(undef, N, N)

    # DC component
    F[:, 1] .= 1 / sqrt(N)

    k = 2
    for m in 1:(N÷2-1)
        F[:, k] .= sqrt(2 / N) .* cos.(2π * m .* t)
        k += 1
        F[:, k] .= sqrt(2 / N) .* sin.(2π * m .* t)
        k += 1
    end

    # Nyquist frequency (only cosine, since sine vanishes)
    F[:, k] .= (N % 2 == 0) ? (sqrt(1 / N) .* cos.(π .* (0:N-1))) : zeros(N)

    return F
end
#This functions makes puts some boundary into the elements of the edges of an array
function make_boundary(a::AbstractVector, n::Int=128)
    A = zeros(n + 1, n + 1)
    A[1:n+1, 1] = a[1:n+1]
    A[n+1, 2:n+1] = a[n+2:2*n+1]
    A[n:-1:1, n+1] = a[2*n+2:3*n+1]
    A[1, n:-1:2] = a[3*n+2:4*n]
    A
end
# Then this function gives a function that interpolates over the array:
function interpolate_array_2D(arr::Array{Float64,2})
    # Ensure the input array is n x n
    @assert size(arr, 1) == size(arr, 2) "The input array must be square (n x n)."

    # Define the range of the original array indices
    n = size(arr, 1)
    xs = 1:n
    ys = 1:n

    # Create an interpolation object
    itp = Interpolations.interpolate((xs, ys), arr, Gridded(Linear()))

    # Define a function to map the interval [-1, 1] to the array index range [1, n]
    return x -> itp(1 + (0.5 * x[1] + 0.5) * (n - 1), 1 + (0.5 * x[2] + 0.5) * (n - 1))
end

@testset "SolverTests" begin
    # Your test code here
    n = 128
    grid = generate_grid(Quadrilateral, (n, n))
    ∂Ω = union(getfacetset.((grid,), ["left", "top", "right", "bottom"])...)
    fe = FerriteFESpace{RefQuadrilateral}(grid, 1, 2, ∂Ω)
    G_full = real_fourier_basis(9)
    rhs_dict = Dict()
    Threads.@threads for i in 2:512
        M = make_boundary(G_full[:, i])
        itp = interpolate_array_2D(M)
        rhs_dict[i] = assemble_rhs_func(fe, itp)
    end
    img = load("SolverTests/Reference128.png")
    itp = interpolate_array_2D(Float64.(img))
    cond_vec = project_function_to_fem(fe, itp)
    K = assemble_L(fe, cond_vec)
    K_fac = factorize(K)
    mode_dict = Dict()
    @time begin
        Threads.@threads for i in 2:512
            mode_dict[i-1] = create_mode_from_g(fe, rhs_dict[i], K)
        end
    end

    sol = FerriteSolverState(fe, cond_vec)
    @test true

    prblm = FerriteProblem(fe, mode_dict, sol)


    @time solve_modes!(prblm, 100, state_adjoint_step_neumann_init!)

    @test true
end
