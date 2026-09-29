# Conformal maps from the unit disk onto star-shaped domains by Theodorsen's method (wiki:
# Numerical Conformal Mapping, Conformal Invariance of the Conductivity Equation).
#
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

"""
    ConformalMap(boundary; center = nothing, modes = 256, tol = 1e-13, maxiter = 2000, relaxation = 1)

Conformal map `Φ` from the unit disk onto the domain bounded by `boundary`, normalised by
`Φ(0) = center` and `Φ'(0) > 0`. `boundary` is a closed curve `t ↦ (x, y)` on `[0, 2π)` or a
vector of polygon vertices; the domain must be star-shaped with respect to `center` (default: its
centroid) and not too far from a disk (Theodorsen's method converges for `|d log ρ/dϑ| < 1` in
polar coordinates about the centre; otherwise an `ArgumentError` is thrown). `modes` Fourier
modes resolve the boundary correspondence; corners need many.

Evaluate with `Φ(w)` (complex `w`, `|w| ≤ 1`) and [`map_derivative`](@ref)`(Φ, w)`. Fields:
`center`, `coefficients` (Taylor coefficients of `log((Φ(w) - center)/w)`), `boundary_error`
(largest relative radial deviation of `Φ(e^{iθ})` from the boundary between the nodes),
`iterations`.
"""
struct ConformalMap
    center::ComplexF64
    coefficients::Vector{ComplexF64}
    boundary_error::Float64
    iterations::Int
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

function ConformalMap(boundary; center = nothing, modes::Integer = 256, tol::Real = 1e-13,
                      maxiter::Integer = 2000, relaxation::Real = 1.0)
    iseven(modes) && modes >= 8 || throw(ArgumentError("modes must be even and ≥ 8"))
    γ = boundary isa AbstractVector ? _polygon_curve(boundary) : (t -> _as_complex(boundary(t)))
    z0 = center === nothing ? _centroid_of(γ) : _as_complex(center)
    curve = _PolarCurve(γ, z0; samples = max(4096, 16modes))
    N = modes
    θ = 2π .* (0:(N - 1)) ./ N
    k = [j <= N ÷ 2 ? j : j - N for j in 0:(N - 1)]
    conj_mult = [-im * sign(kk) * (abs(kk) != N ÷ 2) for kk in k]  # 𝒦 in Fourier space
    ϑ = collect(θ)
    it, δ = 0, Inf
    while it < maxiter
        it += 1
        g = log.(_rho.(Ref(curve), ϑ))
        ϑnew = θ .+ real.(AbstractFFTs.ifft(conj_mult .* AbstractFFTs.fft(g)))
        δ = maximum(abs, ϑnew .- ϑ)
        (isfinite(δ) && δ < 10) ||
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
    Φ = ConformalMap(z0, a, NaN, it)
    # boundary error between the nodes: radial deviation of Φ(e^{iθ}) from ρ at its polar angle
    err = maximum(range(0, 2π; length = 4N + 1)[1:end - 1] .+ π / (4N)) do t
        z = Φ(cis(t)) - z0
        abs(log(abs(z)) - log(_rho(curve, angle(z))))
    end
    return ConformalMap(z0, a, err, it)
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
