using Statistics



export FerriteEITMode
export create_mode_from_g


function mean_boundary!(vec, mode, down)
    mode.b = down(vec)
    mean = Statistics.mean(mode.b)
    mode.b .-= mean
    vec .-= mean
end


function create_mode_from_g(fe::FerriteFESpace, g_vec::AbstractVector, K::AbstractMatrix)
    if length(g_vec) == fe.n
        G = copy(g_vec)
        g = fe.down(G)
        f = fe.down(K \ g_vec)
    elseif length(g_vec) == fe.m
        g = copy(g_vec)
        G = fe.up(g)
        f = fe.down(K \ g_vec)
    end
    u = zeros(fe.n)
    u_g = zeros(fe.n)
    w = zeros(fe.n)
    b = zeros(fe.n)
    λ = zeros(fe.n)
    δσ = zeros(fe.n)

    f = fe.down(K \ g_vec)
    F = fe.up(f)
    λrhs = zeros(fe.n)
    rhs = zeros(fe.n)
    error_d = 0.0
    error_n = 0.0
    error_m = 0.0

    FerriteEITMode(u, u_g, w, b, λ, δσ, F, f, G, g, λrhs, rhs, error_d, error_n, error_m)
end
