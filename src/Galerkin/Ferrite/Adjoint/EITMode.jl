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

function create_mode_from_g(fe::FerriteFESpace, g_vec::AbstractVector, K; σ_g::Number = 0.0, σ_f::Number = 0.0, normalize::Bool = true)
    if length(g_vec) == fe.n
        G = copy(g_vec)
        g = fe.down(G)
        mean_g = Statistics.mean(g)
        G .-= mean_g
        g .-= mean_g
        if normalize
            norm_g = norm(g)^2
            G ./= norm_g
            g ./= norm_g
            mean_g = Statistics.mean(g)
            G .-= mean_g
            g .-= mean_g
        end
        if σ_g ≠ 0
            rand_vec = σ_f * randn(fe.m)
            rand_mean = Statistics.mean(rand_vec)
            rand_vec .-= rand_mean
            g_rand = rand_vec + g
            g_rand .-= Statistics.mean(g_rand)
            f = fe.down(K \ fe.up(g_rand))
        else
            f = fe.down(K \ G)
        end
    elseif length(g_vec) == fe.m
        g = copy(g_vec)
        mean_g = Statistics.mean(g)
        g .-= mean_g
        if normalize
            norm_g = norm(g)^2
            g ./= norm_g
            mean_g = Statistics.mean(g)
            g .-= mean_g
        end
        G = fe.up(g)
        if σ_g ≠ 0
            rand_vec = σ_f * randn(fe.m)
            rand_mean = Statistics.mean(rand_vec)
            rand_vec .-= rand_mean
            g_rand = rand_vec + g
            g_rand .-= Statistics.mean(g_rand)
            f = fe.down(K \ fe.up(g_rand))
        else
            f = fe.down(K \ G)
        end
    end
    mean_f = Statistics.mean(f)
    f .-= mean_f
    if σ_f ≠ 0.0
        rand_vec = σ_f * randn(fe.m)
        rand_mean = Statistics.mean(rand_vec)
        rand_vec .-= rand_mean
        f .+= rand_vec
    end
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

function svd_on_modes(modes::Dict{Int64,FerriteEITMode},fe::FerriteFESpace)
    n = length(modes)
    G = hcat([mode_dict_no_noise[i].g for i in 1:n]...)
    F = hcat([mode_dict_no_noise[i].f for i in 1:n]...)
    Λ = F * pinv(G)
    V,Σ, U =  svd(Λ)
    Σdiag = Diagonal(Σ[1:n])
    Fnew = V[:,1:n]*Σdiag
    Gnew = U[:,1:n]
    mode_dict_svd = Dict{Int64,FerriteEITMode}()
    @time begin
        Threads.@threads for i in 1:n
            mode_dict_svd[i] = create_mode_from_fg(fe,Fnew[:,i],Gnew[:,i])
        end
    end
    mode_dict_svd, Σ[1:n]
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

# For later use:

# Only to be applied to the g or G vector! (or f, F is also ok)
function mean_nonzero!(x::AbstractVector)
    s = zero(eltype(x))
    n = 0
    @inbounds for xi in x
        if xi != 0
            s += xi
            n += 1
        end
    end
    n == 0 && return x  # nothing to do
    μ = s / n
    @inbounds for i in eachindex(x)
        if x[i] != 0
            x[i] -= μ
        end
    end
    return x
end
