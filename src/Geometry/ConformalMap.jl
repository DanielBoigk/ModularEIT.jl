# Conformal maps from the unit disk onto simply connected domains (wiki: Numerical Conformal
# Mapping, Conformal Invariance of the Conductivity Equation): Theodorsen's method for nearly
# circular star-shaped domains, Wegmann's method for general smooth Jordan domains.
#
# Theodorsen:
# The boundary is written in polar form about a centre z₀: z = z₀ + ρ(ϑ) e^{iϑ}. The Riemann map
# Φ: 𝔻 → Ω with Φ(0) = z₀, Φ'(0) > 0 is Φ(w) = z₀ + w exp(F(w)) with F analytic in the disk and
# on the unit circle
#     Re F(e^{iθ}) = log ρ(ϑ(θ)),   Im F(e^{iθ}) = ϑ(θ) - θ,
# where ϑ(θ) is the (unknown) boundary correspondence. Since Im F is the conjugate function of
# Re F, ϑ solves Theodorsen's integral equation
#     ϑ = θ + 𝒦[log ρ ∘ ϑ],     𝒦: e^{ikθ} ↦ -i sign(k) e^{ikθ}  (conjugation, one FFT pair),
# solved by fixed-point iteration at N equispaced θ. The iteration contracts when the boundary is
# "ε-nearly circular", |d log ρ / dϑ| ≤ ε < 1 (ellipses up to aspect ratio ≈ 2.4). The Taylor
# coefficients of F are the Fourier coefficients of log ρ ∘ ϑ, so Φ is analytic in the disk by
# construction; the only error is how well the boundary is matched between the nodes.
#
# Wegmann: for a boundary parametrisation η(s) (any smooth Jordan curve, positively oriented), find
# the correspondence S(t) with η(S(t)) - z₀ = e^{it} G(e^{it}), G analytic, G(0) > 0. Newton step:
# with y = η(S) - z₀ and d = η'(S), the correction U (real) must make y + d U = e^{it} G, i.e.
#     Im(a G) = b,    a = e^{it}/d,  b = Im(y/d),
# a linear Riemann–Hilbert problem of index 0 (a does not wind: e^{it} and η'(S) both wind once).
# With θ = arg a and the analytic Q = exp(-𝒦θ + iθ) (arg Q = θ), a = |a| e^{𝒦θ} Q, so
#     Im(Q G) = c := b / (|a| e^{𝒦θ})   ⇒   Q G = -𝒦c + i c + λ,   λ ∈ ℝ,
# and λ is fixed by Im G(0) = 0. Then U = Re((e^{it} G - y)/d), S ← S + U. The iteration converges
# quadratically from a reasonable start (the arc-length correspondence, rotated to match Φ'(0) > 0);
# each step costs a few FFTs. The map is stored like Theodorsen's: Φ(w) = z₀ + w exp(F(w)) with the
# Taylor coefficients of F = log G (G has no zeros since Φ is univalent).

"""
    ConformalMap(boundary; method = :auto, center = nothing, modes = nothing, max_modes = 4096,
                 boundary_tol = 1e-8, tol = 1e-13, maxiter = 2000, relaxation = 1,
                 boundary_derivative = nothing)

Conformal map `Φ` from the unit disk onto the simply connected domain bounded by `boundary`,
normalised by `Φ(0) = center` and `Φ'(0) > 0`. `boundary` is a closed curve `t ↦ (x, y)` on
`[0, 2π)` or a vector of polygon vertices; `center` must lie inside (default: the centroid if it
does, otherwise the point farthest from the boundary).

- `method = :theodorsen`: star-shaped domains that are not too far from a disk
  (`|d log ρ/dϑ| < 1` in polar coordinates about the centre); fixed-point iteration.
- `method = :wegmann`: any smooth Jordan domain (e.g. non-star-shaped); started from Symm's
  integral equation and polished by Newton's method (quadratic convergence). Uses
  `boundary_derivative` `t ↦ (x', y')` if given, else central differences. Corners are not
  resolved well.
- `method = :auto`: Theodorsen if it converges, Wegmann otherwise.

`modes` Fourier modes resolve the boundary correspondence. By default (`modes = nothing`) they are
doubled from 256 up to `max_modes` until `boundary_error ≤ boundary_tol`. Domains where `|Φ'|`
varies strongly along the boundary need many: this "crowding" grows exponentially with the
aspect ratio (the tips of a 3:1 ellipse need ~4000 modes), so for elongated domains the disk is
a poor model domain. An `ArgumentError` is thrown when no resolution succeeds; a warning when the
tolerance is not reached.

Evaluate with `Φ(w)` (complex `w`, `|w| ≤ 1`) and [`map_derivative`](@ref)`(Φ, w)`. Fields:
`center`, `coefficients` (Taylor coefficients of `log((Φ(w) - center)/w)`), `boundary_error`
(largest deviation of `Φ(e^{iθ})` from the boundary between the nodes, relative to the size of
the domain), `iterations`, `method`.
"""
struct ConformalMap
    center::ComplexF64
    coefficients::Vector{ComplexF64}
    boundary_error::Float64
    iterations::Int
    method::Symbol
end

# boundary curve in polar form about z₀: ρ(ϑ) by inverting the (monotone) polar angle of γ(t)
struct _PolarCurve{F}
    γ::F                      # t ↦ complex point, 2π-periodic
    z0::ComplexF64
    t::Vector{Float64}        # dense parameter samples
    φ::Vector{Float64}        # unwrapped polar angles at the samples (increasing, span 2π)
end

function _PolarCurve(γ, z0; samples::Int = 4096)
    t = collect(range(0, 2π; length = samples + 1))
    φ = zeros(samples + 1)
    prev = γ(t[1]) - z0
    abs(prev) > 0 || throw(ArgumentError("the centre lies on the boundary"))
    φ[1] = angle(prev)
    for i in 2:(samples + 1)
        cur = γ(t[i]) - z0
        abs(cur) > 0 || throw(ArgumentError("the centre lies on the boundary"))
        φ[i] = φ[i - 1] + angle(cur / prev)
        prev = cur
    end
    span = φ[end] - φ[1]
    if span < 0                                          # clockwise: reverse the orientation
        return _PolarCurve(s -> γ(2π - s), z0; samples)
    end
    (abs(span - 2π) < 1e-6 && all(diff(φ) .> 0)) ||
        throw(ArgumentError("the boundary is not star-shaped with respect to the centre"))
    return _PolarCurve(γ, z0, t, φ)
end

# ρ(ϑ): bisection for γ's parameter with polar angle ϑ in the bracketing sample interval
function _rho(c::_PolarCurve, ϑ::Real)
    φ0 = c.φ[1]
    x = φ0 + mod(ϑ - φ0, 2π)
    i = clamp(searchsortedlast(c.φ, x), 1, length(c.φ) - 1)
    lo, hi = c.t[i], c.t[i + 1]
    ref = c.γ(lo) - c.z0
    angle_at(s) = c.φ[i] + angle((c.γ(s) - c.z0) / ref)
    for _ in 1:80
        mid = (lo + hi) / 2
        angle_at(mid) < x ? (lo = mid) : (hi = mid)
        hi - lo <= 4eps(hi) && break
    end
    return abs(c.γ((lo + hi) / 2) - c.z0)
end

_as_complex(p::Complex) = ComplexF64(p)
_as_complex(p) = complex(Float64(p[1]), Float64(p[2]))

# closed polygon as a curve parametrised by arc length on [0, 2π)
function _polygon_curve(vertices)
    V = [_as_complex(v) for v in vertices]
    length(V) >= 3 || throw(ArgumentError("a polygon needs at least 3 vertices"))
    V[end] == V[1] && pop!(V)
    lens = [abs(V[mod1(k + 1, length(V))] - V[k]) for k in eachindex(V)]
    cum = [0; cumsum(lens)] .* (2π / sum(lens))
    return function (t)
        s = mod(t, 2π)
        k = clamp(searchsortedlast(cum, s), 1, length(V))
        λ = (s - cum[k]) / (cum[k + 1] - cum[k])
        return V[k] + λ * (V[mod1(k + 1, length(V))] - V[k])
    end
end

function _centroid_of(γ; samples = 4096)
    z = [γ(2π * k / samples) for k in 0:(samples - 1)]
    A, C = 0.0, 0.0 + 0im
    for k in eachindex(z)
        a, b = z[k], z[mod1(k + 1, length(z))]
        cr = real(a) * imag(b) - imag(a) * real(b)
        A += cr
        C += (a + b) * cr
    end
    return C / (3A)
end

function ConformalMap(boundary; method::Symbol = :auto, center = nothing, modes = nothing,
                      max_modes::Integer = 4096, boundary_tol::Real = 1e-8, tol::Real = 1e-13,
                      maxiter::Integer = 2000, relaxation::Real = 1.0, boundary_derivative = nothing)
    method in (:auto, :theodorsen, :wegmann) ||
        throw(ArgumentError("method must be :auto, :theodorsen or :wegmann, got :$method"))
    γ = boundary isa AbstractVector ? _polygon_curve(boundary) : (t -> _as_complex(boundary(t)))
    z0 = center === nothing ? _default_center(γ) : _as_complex(center)
    _winding(γ, z0) == 0 && throw(ArgumentError("the centre lies outside the domain"))
    dγ = boundary_derivative === nothing ? (t -> (γ(t + 1e-6) - γ(t - 1e-6)) / 2e-6) :
         (t -> _as_complex(boundary_derivative(t)))
    build(N, m) = _conformal_map(γ, dγ, z0, N, m; tol, maxiter, relaxation)
    if modes !== nothing
        iseven(modes) && modes >= 8 || throw(ArgumentError("modes must be even and ≥ 8"))
        return build(modes, method)
    end
    # adaptive resolution: double the number of modes until the boundary is matched. Theodorsen's
    # convergence condition does not depend on N: once it fails, :auto continues with Wegmann.
    N, best, lasterr = 256, nothing, nothing
    while N <= max_modes
        try
            Φ = build(N, method)
            method === :auto && Φ.method === :wegmann && (method = :wegmann)
            Φ.boundary_error <= boundary_tol && return Φ
            (best === nothing || Φ.boundary_error < best.boundary_error) && (best = Φ)
        catch e
            e isa ArgumentError || rethrow()
            lasterr = e
        end
        N *= 2
    end
    best === nothing && throw(lasterr)
    @warn "ConformalMap: boundary error $(best.boundary_error) > $boundary_tol with $max_modes modes " *
          "(strong crowding or corners); increase max_modes"
    return best
end

function _conformal_map(γ, dγ, z0, N, method; tol, maxiter, relaxation)
    if method !== :wegmann
        try
            return _theodorsen(γ, z0, N; tol, maxiter, relaxation)
        catch e
            (method === :auto && e isa ArgumentError) || rethrow()
        end
    end
    return _wegmann(γ, dγ, z0, N; tol, maxiter)
end

# conjugation operator 𝒦 in Fourier space (N points; the Nyquist mode is dropped)
function _conjugation_multiplier(N)
    k = [j <= N ÷ 2 ? j : j - N for j in 0:(N - 1)]
    return [-im * sign(kk) * (abs(kk) != N ÷ 2) for kk in k]
end
_conjugate(g, mult) = real.(AbstractFFTs.ifft(mult .* AbstractFFTs.fft(g)))

function _theodorsen(γ, z0, N; tol, maxiter, relaxation)
    curve = _PolarCurve(γ, z0; samples = max(4096, 16N))
    θ = 2π .* (0:(N - 1)) ./ N
    conj_mult = _conjugation_multiplier(N)
    ϑ = collect(θ)
    it, δ = 0, Inf
    history = Float64[]
    while it < maxiter
        it += 1
        g = log.(_rho.(Ref(curve), ϑ))
        ϑnew = θ .+ real.(AbstractFFTs.ifft(conj_mult .* AbstractFFTs.fft(g)))
        δ = maximum(abs, ϑnew .- ϑ)
        push!(history, δ)
        # a contraction reduces δ geometrically; no progress over 50 iterations means ε ≥ 1
        stalled = it > 50 && δ > tol && δ > 0.5 * history[it - 50]
        (isfinite(δ) && δ < 10 && !stalled) ||
            throw(ArgumentError("Theodorsen's method diverges: the boundary is too far from a disk " *
                                "(|d log ρ/dϑ| ≥ 1) or not star-shaped with respect to the centre"))
        ϑ .+= relaxation .* (ϑnew .- ϑ)
        δ <= tol && break
    end
    δ <= tol || throw(ArgumentError("Theodorsen's method did not converge in $maxiter iterations " *
                                    "(boundary too far from a disk; try relaxation < 1)"))
    all(diff(ϑ) .> 0) || throw(ArgumentError("the boundary correspondence is not monotone: increase modes"))
    g = log.(_rho.(Ref(curve), ϑ))
    ĝ = AbstractFFTs.fft(g) ./ N
    a = ComplexF64[ĝ[1]; 2 .* ĝ[2:(N ÷ 2)]]
    Φ = ConformalMap(z0, a, NaN, it, :theodorsen)
    # boundary error between the nodes: radial deviation of Φ(e^{iθ}) from ρ at its polar angle
    err = maximum(range(0, 2π; length = 4N + 1)[1:end - 1] .+ π / (4N)) do t
        z = Φ(cis(t)) - z0
        abs(log(abs(z)) - log(_rho(curve, angle(z))))
    end
    return ConformalMap(z0, a, err, it, :theodorsen)
end

# S(t) on N equispaced t → on M equispaced t (trigonometric interpolation of the periodic S - t)
function _interpolate_correspondence(S, M)
    N = length(S)
    t = 2π .* (0:(N - 1)) ./ N
    Ŝ = AbstractFFTs.fft(S .- t) ./ N
    P = zeros(ComplexF64, M)
    for j in 0:(N - 1)
        k = j <= N ÷ 2 ? j : j - N
        abs(k) == N ÷ 2 && continue
        P[mod(k, M) + 1] = Ŝ[j + 1]
    end
    return real.(AbstractFFTs.ifft(P)) .* M .+ 2π .* (0:(M - 1)) ./ M
end

function _wegmann(γ, dγ, z0, N; tol, maxiter, oversample::Int = 4)
    # positive orientation
    if _winding(γ, z0) < 0
        γ0, dγ0 = γ, dγ
        γ, dγ = (s -> γ0(-s)), (s -> -dγ0(-s))
    end
    # The Riemann–Hilbert step needs arg(e^{it}/η'(S)) resolved, which varies much faster than the
    # map itself where the correspondence is compressed; Newton therefore runs on an oversampled
    # grid, started from Symm's solution on `N` points interpolated spectrally.
    S = _interpolate_correspondence(_symm_start(γ, dγ, z0, N), oversample * N)
    N = oversample * N
    _increasing(S) || throw(ArgumentError("Wegmann's method: the correspondence is under-resolved " *
                                          "(crowding); increase modes"))
    t = 2π .* (0:(N - 1)) ./ N
    conj_mult = _conjugation_multiplier(N)
    kgrid = [j <= N ÷ 2 ? j : j - N for j in 0:(N - 1)]
    it, δ, δprev = 0, Inf, Inf
    while it < maxiter && δ > tol
        # round-off floor: stop when the corrections stagnate at a tiny level
        δ < 1e-8 && δ > 0.5δprev && break
        δprev = δ
        it += 1
        y, d = γ.(S) .- z0, dγ.(S)
        any(iszero, d) && throw(ArgumentError("the boundary parametrisation has a vanishing derivative"))
        a = cis.(t) ./ d
        θ = _unwrap(angle.(a))
        abs(θ[end] - θ[1]) < π || throw(ArgumentError("Wegmann's method: the boundary correspondence is not monotone"))
        Kθ = _conjugate(θ, conj_mult)
        c = imag.(y ./ d) ./ (abs.(a) .* exp.(Kθ))
        φ, mc = sum(θ) / N, sum(c) / N
        abs(sin(φ)) > 1e-8 || throw(ArgumentError("Wegmann's method: degenerate normalisation"))
        λ = mc * cos(φ) / sin(φ)                             # Im G(0) = 0
        Q = exp.(complex.(-Kθ, θ))
        G = complex.(λ .- _conjugate(c, conj_mult), c) ./ Q
        U = real.((cis.(t) .* G .- y) ./ d)
        # The discrete Newton map amplifies high-frequency round-off and aliasing (errors grow
        # ~50× per step without a filter); keep only the lower half of the oversampled spectrum.
        Û = AbstractFFTs.fft(U)
        Û[abs.(kgrid) .> N ÷ 4] .= 0
        U = real.(AbstractFFTs.ifft(Û))
        # damped step: keep the correspondence increasing
        step = 1.0
        while step > 1e-6 && !_increasing(S .+ step .* U)
            step /= 2
        end
        step > 1e-6 || throw(ArgumentError("Wegmann's method: the boundary correspondence is not monotone; " *
                                            "increase modes or choose another centre"))
        S .+= step .* U
        δ = step * maximum(abs, U)
        (isfinite(δ) && δ < 10) || throw(ArgumentError("Wegmann's method diverges"))
    end
    δ <= max(tol, 1e-8) || throw(ArgumentError("Wegmann's method did not converge in $maxiter iterations"))
    # F = log G from the boundary values of G = (η(S) - z₀) e^{-it} (does not wind)
    Gb = (γ.(S) .- z0) .* cis.(-t)
    F = complex.(log.(abs.(Gb)), _unwrap(angle.(Gb)))
    F̂ = AbstractFFTs.fft(F) ./ N
    a = ComplexF64[F̂[1]; F̂[2:(N ÷ 2)]]
    a[1] = real(a[1])                                        # Φ'(0) > 0
    Φ = ConformalMap(z0, a, NaN, it, :wegmann)
    # boundary error between the nodes: |Φ(e^{iτ}) - η(S(τ))| with S - t interpolated spectrally
    Ŝ = AbstractFFTs.fft(S .- t)
    k = [j <= N ÷ 2 ? j : j - N for j in 0:(N - 1)]
    shift = π / N
    Smid = real.(AbstractFFTs.ifft(Ŝ .* cis.(k .* shift) .* (abs.(k) .!= N ÷ 2))) .+ t .+ shift
    diam = maximum(abs, γ.(S) .- z0)
    err = maximum(abs(Φ(cis(τ)) - γ(s)) for (τ, s) in zip(t .+ shift, Smid)) / diam
    return ConformalMap(z0, a, err, it, :wegmann)
end

# correspondence S(t) - t periodic and S strictly increasing
_increasing(S) = all(diff(S) .> 0) && S[1] + 2π > S[end]

function _unwrap(φ)
    out = copy(φ)
    for i in 2:length(out)
        out[i] = out[i - 1] + mod(φ[i] - φ[i - 1] + π, 2π) - π
    end
    return out
end

# winding number of the curve γ about z (dense sampling)
function _winding(γ, z; samples = 4096)
    total = 0.0
    prev = γ(0.0) - z
    for k in 1:samples
        cur = γ(2π * k / samples) - z
        total += angle(cur / prev)
        prev = cur
    end
    return round(Int, total / 2π)
end

# Starting correspondence from Symm's integral equation. The harmonic measure density of Ω seen
# from z₀, as a density w(τ) in the curve parameter, solves the first-kind equation
#     ∫ log|η(s) - η(τ)| w(τ) dτ + C = log|η(s) - z₀|,   ∫ w dτ = 1
# (C = 0 exactly; the free constant keeps the bordered system nonsingular also when the
# logarithmic capacity of Γ is 1), and the boundary correspondence is t(s) = t₀ + 2π ∫₀ˢ w.
# Nyström discretisation with Kress's quadrature for the logarithmic singularity (spectral for
# smooth curves); t(s) is then inverted and rotated so that Φ'(0) > 0.
function _symm_start(γ, dγ, z0, N; dense = 16)
    s = 2π .* (0:(N - 1)) ./ N
    η, dη = γ.(s), dγ.(s)
    R = _kress_weights(N)
    A = zeros(N + 1, N + 1)
    for j in 1:N, i in 1:N
        smooth = i == j ? log(abs(dη[i])) : log(abs(η[i] - η[j])) - log(abs(2sin((s[i] - s[j]) / 2)))
        A[i, j] = R[mod(i - j, N) + 1] + 2π / N * smooth
    end
    A[1:N, N + 1] .= 1
    A[N + 1, 1:N] .= 2π / N
    w = (A \ [log.(abs.(η .- z0)); 1.0])[1:N]
    all(>(0), w) || throw(ArgumentError("Symm's equation: non-positive harmonic measure density; increase modes"))
    # t(s) = s + 2π ∫₀ˢ (w - 1/2π) on a dense grid (spectral integration of the oscillating part)
    ŵ = AbstractFFTs.fft(w .- 1 / 2π) ./ N
    k = [j <= N ÷ 2 ? j : j - N for j in 0:(N - 1)]
    M = dense * N
    sd = 2π .* (0:M) ./ M
    integral(x) = real(sum(ŵ[m] * (cis(k[m] * x) - 1) / (im * k[m]) for m in 2:N if abs(k[m]) != N ÷ 2))
    td = [x + 2π * integral(x) for x in sd]
    all(diff(td) .> 0) || throw(ArgumentError("Symm's equation: the correspondence is not monotone; increase modes"))
    # invert t(s) at τ + α (α: rotation), linearly on the dense grid
    Sof(τ) = (x = td[1] + mod(τ - td[1], 2π); i = clamp(searchsortedlast(td, x), 1, M);
              sd[i] + (x - td[i]) / (td[i + 1] - td[i]) * (sd[i + 1] - sd[i]))
    t = s
    S0 = Sof.(t)
    m = sum((γ.(S0) .- z0) .* cis.(-t)) / N              # ≈ Φ'(0) for the unrotated map
    S = Sof.(t .- angle(m))
    for i in 2:N                                        # increasing representative
        while S[i] <= S[i - 1]
            S[i] += 2π
        end
    end
    S .-= 2π * floor((S[1] - t[1] + π) / 2π)
    return S
end

# Kress's quadrature weights: ∫₀^{2π} log|2 sin((s-τ)/2)| φ(τ) dτ ≈ Σⱼ R[(i - j) mod N] φ(τⱼ) at the
# nodes s = τᵢ, exact for trigonometric interpolants of degree N/2
function _kress_weights(N)
    n = N ÷ 2
    return [-(2π / N) * (sum(cos(m * x) / m for m in 1:(n - 1); init = 0.0) + cos(n * x) / (2n))
            for x in 2π .* (0:(N - 1)) ./ N]
end

# default centre: the centroid if it lies inside, else the sampled point farthest from the boundary
function _default_center(γ)
    c = _centroid_of(γ)
    _winding(γ, c) != 0 && return c
    z = γ.(range(0, 2π; length = 2049)[1:end - 1])
    xs, ys = extrema(real.(z)), extrema(imag.(z))
    best, dbest = c, -Inf
    for x in range(xs...; length = 64), y in range(ys...; length = 64)
        p = complex(x, y)
        _winding(γ, p; samples = 512) == 0 && continue
        d = minimum(abs.(z .- p))
        d > dbest && ((best, dbest) = (p, d))
    end
    isfinite(dbest) || throw(ArgumentError("no interior point found; pass center"))
    return best
end

# F(w) = Σ aₖ wᵏ and F'(w) by Horner
function _taylor(Φ::ConformalMap, w)
    a = Φ.coefficients
    F, dF = a[end], zero(ComplexF64)
    for k in (length(a) - 1):-1:1
        dF = dF * w + F
        F = F * w + a[k]
    end
    return F, dF
end

(Φ::ConformalMap)(w::Number) = (F = first(_taylor(Φ, w)); Φ.center + w * exp(F))

"""
    map_derivative(Φ::ConformalMap, w)

Derivative `Φ'(w)` of the conformal map; `|Φ'|` is the local length scale factor (boundary
current densities and contact impedances transform with it).
"""
function map_derivative(Φ::ConformalMap, w::Number)
    F, dF = _taylor(Φ, w)
    return exp(F) * (1 + w * dF)
end
