using Statistics



export FerriteEITMode


mutable struct FerriteEITMode
    u_f::Union{AbstractVector,Nothing}
    u_g::Union{AbstractVector,Nothing}
    w::Union{AbstractVector,Nothing}
    b::Union{AbstractVector,Nothing}
    λ::AbstractVector
    δσ::AbstractVector
    F::Union{AbstractVector,Nothing} # This is the long vector for dirichlet boundary conditions
    f::Union{AbstractVector,Nothing} # This is the short vector for dirichlet boundary conditions
    G::Union{AbstractVector,Nothing} # This is the long vector for neumann boundary conditions
    g::Union{AbstractVector,Nothing} # This is the short vector for neumann boundary conditions
    λrhs::AbstractVector
    rhs::AbstractVector # This is a preallocation for calculating the bilinear map
    error_d::Float64
    error_n::Float64
    error_m::Float64
end

function mean_boundary!(vec, mode, down)
    mode.b = down(vec)
    mean = Statistics.mean(mode.b)
    mode.b .-= mean
    vec .-= mean
end
