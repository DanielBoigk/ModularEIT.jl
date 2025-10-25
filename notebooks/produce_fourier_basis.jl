using Interpolations
using Statistics
# This is for the test run were only norm
function make_512_boundary(a::AbstractVector)
    A = zeros(129,129)
    A[1:129,1] = a[1:129]
    A[129,2:129] = a[130:130+127]
    A[128:-1:1,129] = a[258:258+127]
    A[1, 128:-1:2] = a[386:512]
    A
end

function real_fourier_basis(n::Int)
    N = 2^n
    t = collect(0:N-1) ./ N
    F = Matrix{Float64}(undef, N, N)

    # DC component
    F[:, 1] .= 1 / sqrt(N)

    k = 2
    for m in 1:(N÷2 - 1)
        F[:, k]   .= sqrt(2/N) .* cos.(2π * m .* t); k += 1
        F[:, k]   .= sqrt(2/N) .* sin.(2π * m .* t); k += 1
    end

    # Nyquist frequency (only cosine, since sine vanishes)
    F[:, k] .= (N % 2 == 0) ? (sqrt(1/N) .* cos.(π .* (0:N-1))) : zeros(N)

    mean_F = mean(F, dims=2)
    F = F .- mean_F
    return F
end
