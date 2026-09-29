# Rotationally symmetric triangle meshes of the disk and detection of their polar structure for
# the FFT solver (wiki: Fast Solvers on Disk Domains).

"""
    polar_grid(nr, nθ; radius = 1, boundary_spacing = nothing, offset = 0)

Triangle mesh of the regular `nθ`-gon inscribed in the circle of the given `radius`: a centre
node and `nr` rings of `nθ` nodes at the angles `offset + 2π j / nθ`. The centre is connected to
the first ring by a fan, and every ring-to-ring quadrilateral is split along the same diagonal,
so the mesh is invariant under the rotation by `2π/nθ` (as required by the FFT solver, see
[`PolarPreconditioner`](@ref)). Rings are uniformly spaced, or geometrically graded towards the
boundary with the outermost spacing `boundary_spacing` (EIT sensitivity and the electrode edges
are concentrated at the boundary). For `L` electrodes from [`angular_electrodes`](@ref) with
`coverage` c, choose `nθ` as a multiple of `2L/c` (e.g. `4L` for `c = 1/2`): then the electrode
edges lie on nodes and all electrodes consist of the same number of boundary facets.
"""
function polar_grid(nr::Integer, nθ::Integer; radius::Real = 1.0, boundary_spacing = nothing, offset::Real = 0.0)
    nr >= 1 && nθ >= 3 || throw(ArgumentError("need nr ≥ 1 rings and nθ ≥ 3 nodes per ring"))
    radii = _ring_radii(nr, Float64(radius), boundary_spacing)
    nodes = [Node(Vec(0.0, 0.0))]
    for r in radii, j in 0:(nθ - 1)
        θ = offset + 2π * j / nθ
        push!(nodes, Node(Vec(r * cos(θ), r * sin(θ))))
    end
    id(k, j) = 1 + (k - 1) * nθ + mod(j, nθ) + 1
    cells = Triangle[]
    for j in 0:(nθ - 1)
        push!(cells, Triangle((1, id(1, j), id(1, j + 1))))
    end
    for k in 1:(nr - 1), j in 0:(nθ - 1)
        a, b, c, d = id(k, j), id(k + 1, j), id(k + 1, j + 1), id(k, j + 1)
        push!(cells, Triangle((a, b, c)), Triangle((a, c, d)))
    end
    return Grid(cells, nodes)
end

# ring radii r₁ < … < r_nr = R: uniform, or geometric spacings Δₖ = Δ_nr q^(nr-k) with q ≥ 1
function _ring_radii(nr, R, s)
    s === nothing && return R .* (1:nr) ./ nr
    0 < s <= R / nr || throw(ArgumentError("boundary_spacing must lie in (0, radius/nr] (spacings grow inwards)"))
    total(q) = s * sum(q^i for i in 0:(nr - 1))
    lo, hi = 1.0, 2.0
    while total(hi) < R
        hi *= 2
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        total(mid) < R ? (lo = mid) : (hi = mid)
    end
    q = (lo + hi) / 2
    Δ = [s * q^(nr - k) for k in 1:nr]
    r = cumsum(Δ)
    return r .* (R / r[end])
end

"""
    polar_structure(disc; rtol = 1e-8)

[`PolarStructure`](@ref) of a discretization with linear triangles (P1 u space) on a
rotationally symmetric disk mesh such as [`polar_grid`](@ref): the centre node, rings of equal
node counts at uniformly spaced angles, and a stiffness matrix invariant under the rotation.
Throws an `ArgumentError` otherwise.
"""
function polar_structure(d::FerriteDiscretization; rtol::Real = 1e-8)
    fail(msg) = throw(ArgumentError("not a rotationally symmetric disk mesh: $msg"))
    d.ip_u isa Lagrange{RefTriangle, 1} || fail("the u space must be linear triangles (P1)")
    d.C_u === nothing || fail("hanging nodes are not supported")
    n = ndofs_u(d)
    xs, ys = _dof_coordinates(d, 1:n)
    bx, by = _dof_coordinates(d, d.boundary_dofs)
    cx, cy = sum(bx) / length(bx), sum(by) / length(by)
    r = hypot.(xs .- cx, ys .- cy)
    θ = atan.(ys .- cy, xs .- cx)
    R = maximum(r)
    tol = rtol * R
    centre = findall(<=(tol), r)
    length(centre) == 1 || fail("need exactly one centre node")
    others = setdiff(1:n, centre)
    radii = _distinct(r[others], tol)
    nr = length(radii)
    nθ, rem_ = divrem(length(others), nr)
    rem_ == 0 || fail("rings have different node counts")
    perm = zeros(Int, n)
    perm[1] = centre[1]
    Δθ = 2π / nθ
    for (k, ρ) in enumerate(radii)
        ring = [i for i in others if abs(r[i] - ρ) <= tol]
        length(ring) == nθ || fail("ring $k has $(length(ring)) nodes, expected $nθ")
        θ0 = minimum(mod(θ[i], Δθ) for i in ring)
        for i in ring
            t = (mod(θ[i], 2π) - θ0) / Δθ
            j = round(Int, t)
            abs(t - j) <= rtol * nθ || fail("ring $k is not uniformly spaced")
            p = 1 + (k - 1) * nθ + mod(j, nθ) + 1
            perm[p] == 0 || fail("two nodes at the same angle in ring $k")
            perm[p] = i
        end
    end
    K = assemble_stiffness(d.dh_u, d.cv_u)
    return PolarStructure(nr, nθ, true, perm, K[perm, perm])
end
