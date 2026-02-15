using Interpolations

export real_fourier_basis, make_boundary, interpolate_array_2D

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

function equidistant_grid(n::Integer)
    xs = range(-1, 1; length=n)
    [Vec(x, y) for x in xs, y in xs]
end
