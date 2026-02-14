using Statistics
using LinearAlgebra


export FerriteEITMode
export create_mode_from_g, create_mode_from_fg, create_mode_from_f
export add_noise_f!, add_noise_g!
#export svd

function mean_boundary!(vec, mode, down)
    mode.b = down(vec)
    mean = Statistics.mean(mode.b)
    mode.b .-= mean
    vec .-= mean
end

function create_mode_from_g(fe::FerriteFESpace, g_vec::AbstractVector, K)
    if length(g_vec) == fe.n
        G = copy(g_vec)
        g = fe.down(G)
        mean_g = Statistics.mean(g)
        G .-= mean_g
        g .-= mean_g
        f = fe.down(K \ G)
    elseif length(g_vec) == fe.m
        g = copy(g_vec)
        mean_g = Statistics.mean(g)
        g .-= mean_g
        G = fe.up(g)
        f = fe.down(K \ G)
    end
    mean_f = Statistics.mean(f)
    f .-= mean_f
    u = zeros(fe.n)
    u_g = zeros(fe.n)
    w = zeros(fe.n)
    b = zeros(fe.m)
    λ = zeros(fe.n)
    δσ = zeros(fe.n)

    F = fe.up(f)
    λrhs = zeros(fe.n)
    rhs = zeros(fe.n)
    error_d = 0.0
    error_n = 0.0
    error_m = 0.0

    FerriteEITMode(u, u_g, w, b, λ, δσ, F, f, G, g, λrhs, rhs, error_d, error_n, error_m)
end

function create_mode_from_f(fe::FerriteFESpace, f_vec::AbstractVector, KD, KN)
    if length(f_vec) == fe.n
        F = copy(f_vec)
        f = fe.down(F)
        mean_f = Statistics.mean(f)
        f .-= mean_f
        F .-= mean_f
        u_true = KD \ F
    elseif length(f_vec) == fe.m
        f = copy(f_vec)
        mean_f = Statistics.mean(f)
        f .-= mean_f
        F = fe.up(f)
        u_true = KD \ F
    end
    g = fe.down(KN * u_true)
    mean_g = Statistics.mean(g)
    g .-= mean_g
    u = zeros(fe.n)
    u_g = zeros(fe.n)
    w = zeros(fe.n)
    b = zeros(fe.m)
    λ = zeros(fe.n)
    δσ = zeros(fe.n)

    G = fe.up(g)
    λrhs = zeros(fe.n)
    rhs = zeros(fe.n)
    error_d = 0.0
    error_n = 0.0
    error_m = 0.0

    FerriteEITMode(u, u_g, w, b, λ, δσ, F, f, G, g, λrhs, rhs, error_d, error_n, error_m)
end

function create_mode_from_fg(fe::FerriteFESpace, f_vec::AbstractVector, g_vec::AbstractVector)
    if length(f_vec) == fe.n
        F = copy(f_vec)
        f = fe.down(F)
        mean_f = Statistics.mean(f)
        f .-= mean_f
        F .-= mean_f
    elseif length(f_vec) == fe.m
        f = copy(f_vec)
        mean_f = Statistics.mean(f)
        f .-= mean_f
        F = fe.up(f)
    end
    if length(g_vec) == fe.n
        G = copy(g_vec)
        g = fe.down(G)
        mean_g = Statistics.mean(g)
        g .-= mean_g
        G .-= mean_g
    elseif length(g_vec) == fe.m
        g = copy(g_vec)
        mean_g = Statistics.mean(g)
        g .-= mean_g
        G = fe.up(g)
    end

    u = zeros(fe.n)
    u_g = zeros(fe.n)
    w = zeros(fe.n)
    b = zeros(fe.m)
    λ = zeros(fe.n)
    δσ = zeros(fe.n)

    λrhs = zeros(fe.n)
    rhs = zeros(fe.n)
    error_d = 0.0
    error_n = 0.0
    error_m = 0.0

    FerriteEITMode(u, u_g, w, b, λ, δσ, F, f, G, g, λrhs, rhs, error_d, error_n, error_m)
end

function add_noise_f!(mode::FerriteEITMode, noise_vec::AbstractVector, fe::FerriteFESpace)
    noise_mean = Statistics.mean(noise_vec)
    @. mode.f += noise_vec - noise_mean
    mode.F = fe.up(mode.f)
    nothing
end
function add_noise_f!(mode::FerriteEITMode, n::Int, σ::Real, fe::FerriteFESpace)
    noise_vec = σ * randn(n)
    add_noise_f!(mode, noise_vec, fe)
end

function add_noise_g!(mode::FerriteEITMode, noise_vec::AbstractVector, fe::FerriteFESpace)
    noise_mean = Statistics.mean(noise_vec)
    @. mode.g += noise_vec - noise_mean
    mode.G = fe.up(mode.g)
    nothing
end
function add_noise_g!(mode::FerriteEITMode, n::Int, σ::Real, fe::FerriteFESpace)
    noise_vec = σ * randn(n)
    add_noise_g!(mode, noise_vec, fe)
end
#=
function svd(modes::Dict{T,FerriteEITMode}, fe::FerriteFESpace) where {T}
    out = Dict{T,FerriteEITMode}()
    # collect g's and f's
    G = [mode.g for mode in values(modes)]
    F = [mode.f for mode in values(modes)]
    G = hcat(G...)
    F = hcat(F...)
    Λ = G * F'
    U, Σ, V = LinearAlgebra.svd(Λ)
    Σ = Σ[Σ.>1e-10]
    num_modes = length(Σ)

    for i in 1:num_modes
        out[i] = create_mode_from_fg(fe, Σ[i] * U[:, i], V[:, i])
    end
    out, num_modes
end
=#
