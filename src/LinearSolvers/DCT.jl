# Fast solver for the constant-coefficient Neumann problem on a uniform rectangle grid of bilinear
# (Q1) elements, by the discrete cosine transform (wiki: Discrete Cosine Transform, Fast Solvers
# on Rectangular Domains).
#
# 1D (N + 1 nodes, spacing h): K V = W V Λ_K, M V = W V Λ_M with the DCT-I modes
# V[j, k] = cos(π j k / N), W = diag(½, 1, …, 1, ½), Vᵀ W V = D = diag(N, N/2, …, N/2, N).
# 2D: K₂ = K_x ⊗ M_y + M_x ⊗ K_y; for X (nx × ny, x index first)
#     K₂(X) = W_x V_x (Λ ∘ C) V_yᵀ W_y   with X = V_x C V_yᵀ,  Λ_kl = λᴷ_k λᴹ_l + λᴹ_k λᴷ_l,
# so K₂⁺ B = V_x [(D_x⁻¹ V_xᵀ B V_y D_y⁻¹) ./ Λ] V_yᵀ with the constant mode (Λ₀₀ = 0) dropped
# (the solution then has zero trapezoidal mean, i.e. ∫ u = 0).
#
# Transforms: T(x) = 2 Vᵀ W x = Re FFT of the even extension (x₀, …, x_N, x_{N-1}, …, x₁), so
# Vᵀ B V = ¼ T_x T_y(W⁻¹ B W⁻¹) and V C Vᵀ = ¼ T_x T_y(W⁻¹ C W⁻¹) (V is symmetric). Only real
# FFTs through AbstractFFTs are used, so the same code runs on GPU arrays whose package provides
# FFT plans.

import AbstractFFTs
import FFTW          # CPU FFT backend for AbstractFFTs

"""
    StructuredGrid

A uniform tensor grid of `nx × ny` nodes with spacings `hx`, `hy`, and the permutation `perm`
from lexicographic node order (x index fastest) to the u dofs of a discretization, with the data
of the transform solver. Built by `structured_grid(disc)`.
"""
struct StructuredGrid
    nx::Int
    ny::Int
    hx::Float64
    hy::Float64
    origin::NTuple{2, Float64}
    perm::Vector{Int}             # lexicographic position → u dof
    wx::Vector{Float64}           # trapezoidal weights (½, 1, …, 1, ½)
    wy::Vector{Float64}
    winv::Matrix{Float64}         # 1 / (wxᵢ wyⱼ)
    sinv::Matrix{Float64}         # scaling of the pseudo-inverse K₂⁺ in the transform basis
    Λ::Matrix{Float64}            # eigenvalues Λ_kl (Λ₀₀ = 0)
    extx::Vector{Int}             # even-extension indices
    exty::Vector{Int}
end

function StructuredGrid(nx::Integer, ny::Integer, hx::Real, hy::Real, origin, perm::AbstractVector{<:Integer})
    nx >= 2 && ny >= 2 || throw(ArgumentError("need at least 2 nodes per direction"))
    Nx, Ny = nx - 1, ny - 1
    θx, θy = π .* (0:Nx) ./ Nx, π .* (0:Ny) ./ Ny
    λK(θ, h) = (2 / h) * (1 - cos(θ))
    λM(θ, h) = (h / 6) * (4 + 2cos(θ))
    Λ = [λK(a, hx) * λM(b, hy) + λM(a, hx) * λK(b, hy) for a in θx, b in θy]
    Λ[1, 1] = 0
    d(N) = [k == 0 || k == N ? Float64(N) : N / 2 for k in 0:N]
    dx, dy = d(Nx), d(Ny)
    wx = [i == 1 || i == nx ? 0.5 : 1.0 for i in 1:nx]
    wy = [j == 1 || j == ny ? 0.5 : 1.0 for j in 1:ny]
    winv = 1 ./ (wx .* wy')
    sinv = winv ./ (16 .* dx .* dy' .* Λ)
    sinv[1, 1] = 0
    ext(N) = [1:(N + 1); N:-1:2]
    return StructuredGrid(nx, ny, hx, hy, Tuple(Float64.(origin)), collect(Int, perm), wx, wy, winv, sinv,
                          Λ, ext(Nx), ext(Ny))
end

# T along dimension `dim` of a 3-array: Re rfft of the even extension (length 2N → N + 1 outputs)
function _dct1(X::AbstractArray{<:Real, 3}, dim::Int, ext::Vector{Int})
    E = dim == 1 ? X[ext, :, :] : X[:, ext, :]
    return real.(AbstractFFTs.rfft(E, dim))
end

function _transform(g::StructuredGrid, X::AbstractArray{<:Real, 3})
    return _dct1(_dct1(X, 1, g.extx), 2, g.exty)
end

"""
    dct_neumann_solve(grid, B)

`K₂⁺ B` for the Q1 Neumann stiffness matrix `K₂` of the structured grid, with `B` (`nx ny × s`,
or a vector) in lexicographic node order: the solution with zero trapezoidal mean (`∫ u = 0`);
`B` must be orthogonal to the constants for `K₂ X = B` to hold.
"""
function dct_neumann_solve(g::StructuredGrid, B::AbstractVecOrMat)
    s = size(B, 2)
    X = reshape(B, g.nx, g.ny, s) .* g.winv
    C = _transform(g, X) .* g.sinv
    Y = _transform(g, C)
    return B isa AbstractVector ? vec(Y) : reshape(Y, g.nx * g.ny, s)
end
