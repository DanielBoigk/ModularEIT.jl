# Adaptive meshing for EIT.
#
# Refinement: Ferrite's AMR (p4est-style forest of quadtrees/octrees) on quadrilateral and
# hexahedral meshes, which fits the pixel/voxel meshes of image-based EIT. Refined meshes have
# hanging nodes; FerriteDiscretization condenses them with conformity constraints, so forward
# models, objectives and solvers work unchanged. Triangle/tetrahedron refinement (e.g. newest
# vertex bisection) is not implemented yet: AdaptiveMesh warns and leaves such meshes unchanged.
#
# Indicators:
#   flux_recovery_indicator  forward accuracy in the energy norm: Zienkiewicz–Zhu estimator of
#                            the current density σ∇u, summed over all current patterns
#   goal_oriented_indicator  accuracy of the measured voltages: ZZ error of the states times ZZ
#                            error of the measurement duals (the adjoint fields of the Jacobian)
#   jump_indicator           reconstruction features: jumps of a piecewise constant σ
# Marking: dorfler_marking. Transfer of σ between meshes: transfer_conductivity.

"""
    AdaptiveMesh(grid; maxlevel = 10)

Adaptively refinable mesh built from a quadrilateral or hexahedral `grid` (forest of
quadtrees/octrees from Ferrite's AMR, at most `maxlevel` refinement levels). Use
[`refine_mesh!`](@ref) / [`coarsen_mesh!`](@ref) and build a new
[`FerriteDiscretization`](@ref) on [`current_grid`](@ref) after each change.

Other cell types (triangles, tetrahedra) are accepted with a warning and are never refined:
refinement for them is not implemented yet.
"""
mutable struct AdaptiveMesh{F}
    forest::F
    grid::Ferrite.AbstractGrid
end

function AdaptiveMesh(grid::Ferrite.AbstractGrid; maxlevel::Integer = 10)
    C = getcelltype(grid)
    if C <: Union{Quadrilateral, Hexahedron}
        forest = Ferrite.AMR.ForestBWG(grid, maxlevel)
        return AdaptiveMesh(forest, Ferrite.AMR.creategrid(forest))
    end
    @warn "Adaptive refinement is implemented for quadrilateral and hexahedral meshes (Ferrite's AMR); " *
          "refinement of $C meshes is not implemented yet, so this mesh will not be refined."
    return AdaptiveMesh(nothing, grid)
end

"""
    current_grid(am::AdaptiveMesh)

The current (possibly non-conforming) grid.
"""
current_grid(am::AdaptiveMesh) = am.grid

_warn_no_refinement() =
    @warn "Adaptive refinement is not implemented for this cell type yet; the mesh is unchanged."

"""
    refine_mesh!(am, cells)
    refine_mesh!(am)

Refine the cells `cells` of the current grid (or all cells) once and restore the 2:1 balance
between neighbouring cells. Cell numbers refer to `current_grid(am)`.
"""
function refine_mesh!(am::AdaptiveMesh, cells::AbstractVector{<:Integer})
    am.forest === nothing && (_warn_no_refinement(); return am)
    isempty(cells) && return am
    Ferrite.AMR.refine!(am.forest, collect(cells))
    Ferrite.AMR.balanceforest!(am.forest)
    am.grid = Ferrite.AMR.creategrid(am.forest)
    return am
end
refine_mesh!(am::AdaptiveMesh) = refine_mesh!(am, collect(1:getncells(am.grid)))

"""
    cell_levels(am::AdaptiveMesh)

Refinement level of every cell of `current_grid(am)` (0 = cell of the initial grid).
"""
function cell_levels(am::AdaptiveMesh)
    am.forest === nothing && return zeros(Int, getncells(am.grid))
    return [Int(leaf.l) for tree in am.forest.cells for leaf in tree.leaves]
end

"""
    max_level(am::AdaptiveMesh)

Maximum refinement level; cells at this level are not refined further. Exclude them from
marking (e.g. set their indicator to zero) so that the adaptive loop keeps making progress.
"""
max_level(am::AdaptiveMesh) = am.forest === nothing ? 0 : Int(first(am.forest.cells).b)

"""
    coarsen_mesh!(am, cells)

Coarsen: every complete family of sibling cells among `cells` is merged into its parent.
"""
function coarsen_mesh!(am::AdaptiveMesh, cells::AbstractVector{<:Integer})
    am.forest === nothing && (_warn_no_refinement(); return am)
    isempty(cells) && return am
    Ferrite.AMR.coarsen!(am.forest, collect(cells))
    Ferrite.AMR.balanceforest!(am.forest)
    am.grid = Ferrite.AMR.creategrid(am.forest)
    return am
end

"""
    flux_recovery_indicator(disc, σ, X; normalize = false)

Zienkiewicz–Zhu error indicator of the current density, one value per cell:

    η_K² = Σₛ ∫_K |J*ₛ - σ ∇uₛ|² dx,

where `J*ₛ` is the L² projection of the discrete current density `σ∇uₛ` onto continuous
linear vector fields. `X` holds the states of all patterns (`n × s`, e.g. from
[`forward_neumann`](@ref); only the first `ndofs_u(disc)` rows, the potential, are used).
Large values mark electrode edges and conductivity interfaces, where the discrete current
density jumps. With `normalize = true` every pattern's indicator is scaled to unit sum before
summing, so that low-energy patterns (high spatial frequency) weigh as much as the dominant
low-frequency ones.
"""
function flux_recovery_indicator(disc::FerriteDiscretization, σ::AbstractVector, X::AbstractVecOrMat;
                                 normalize::Bool = false)
    Xm = _as_matrix(X)
    U = _lift(disc, Xm[1:ndofs_u(disc), :])
    grid = disc.grid
    dim = Ferrite.getspatialdim(grid)
    shape = Ferrite.getrefshape(getcelltype(grid))
    cv_u, cv_σ = disc.cv_u, disc.cv_σ
    qr = cv_u.qr
    nq = getnquadpoints(cv_u)
    ip = Lagrange{shape, 1}()
    proj = L2Projector(ip, grid)
    cv_rec = CellValues(qr, ip^dim)
    ncell = getncells(grid)
    η = zeros(ncell)
    ηs = zeros(ncell)
    flux = [Vector{Vec{dim, Float64}}(undef, nq) for _ in 1:ncell]
    ue = zeros(getnbasefunctions(cv_u))
    σe = zeros(getnbasefunctions(cv_σ))
    σdofs = zeros(Int, getnbasefunctions(cv_σ))
    for s in axes(U, 2)
        u = view(U, :, s)
        for cell in CellIterator(disc.dh_u)
            c = cellid(cell)
            reinit!(cv_u, cell)
            reinit!(cv_σ, cell)
            celldofs!(σdofs, disc.dh_σ, c)
            for (a, d) in enumerate(σdofs)
                σe[a] = σ[d]
            end
            for (i, d) in enumerate(celldofs(cell))
                ue[i] = u[d]
            end
            for q in 1:nq
                flux[c][q] = function_value(cv_σ, q, σe) * function_gradient(cv_u, q, ue)
            end
        end
        J = project(proj, flux, qr)
        fill!(ηs, 0)
        for (c, cell) in enumerate(CellIterator(proj.dh))
            reinit!(cv_rec, cell)
            Je = reinterpret(Float64, J[celldofs(cell)])
            for q in 1:nq
                e = function_value(cv_rec, q, Je) - flux[c][q]
                ηs[c] += (e ⋅ e) * getdetJdV(cv_rec, q)
            end
        end
        tot = sum(ηs)
        η .+= normalize && tot > 0 ? ηs ./ tot : ηs
    end
    return η
end

"""
    goal_oriented_indicator(disc, fm, σ, X; solver = DirectSolver(), normalize = false)

Goal-oriented indicator for the measured voltages, one value per cell:

    η_K = η_K(u) η_K(z),   η_K(u)² = Σₛ ‖J*ₛ - σ∇uₛ‖²_K,   η_K(z)² = Σₘ ‖J*(zₘ) - σ∇zₘ‖²_K,

where `zₘ` are the dual solutions of the measurements (`A zₘ = Qᵀ Π eₘ`, the adjoint fields of
the Jacobian rows) and both factors are [`flux_recovery_indicator`](@ref)s. The voltage error is
bounded by products of primal and dual errors, so cells are refined where errors are made *and*
influence the measurements. `X` holds the current-driven states (`n × s`). `normalize` weighs
every pattern and every measurement equally.

This is a heuristic built from recovery estimators. In `benchmark/adaptive_meshing.jl` it
improves the voltages of low-frequency patterns about 5× over uniform meshes of the same size,
but under-resolves high-frequency patterns (also with `normalize = true`), so the total voltage
error stagnates near 1e-2; a residual-based dual-weighted estimator is the next step.
"""
function goal_oriented_indicator(disc::FerriteDiscretization, fm::ForwardModel, σ::AbstractVector,
                                 X::AbstractVecOrMat; solver::AbstractLinearSolver = DirectSolver(),
                                 normalize::Bool = false)
    system_matrix!(fm, σ)
    st = _init_neumann_solver(solver, fm)
    Πt = _remove_mean!(Matrix(1.0I, n_measure(fm), n_measure(fm)))
    Z = zeros(fm.n, n_measure(fm))
    _solve!(Z, st, fm.Q' * Πt)
    ηu = flux_recovery_indicator(disc, σ, X; normalize)
    ηz = flux_recovery_indicator(disc, σ, Z; normalize)
    return sqrt.(ηu .* ηz)
end

"""
    jump_indicator(disc, σ)

Feature indicator of a piecewise constant conductivity, one value per cell:
`Σ_F |F| |σ_K - σ_K'| / 2` over the facets `F` of the cell. Refining where it is large
resolves inclusion boundaries; cells with zero indicator are candidates for coarsening.
"""
function jump_indicator(disc::FerriteDiscretization, σ::AbstractVector)
    (disc.ip_σ isa DiscontinuousLagrange && Ferrite.getorder(disc.ip_σ) == 0) ||
        throw(ArgumentError("jump_indicator needs a piecewise constant σ"))
    J = zeros(getncells(disc.grid))
    for (c1, f1, c2) in disc.interior_facets
        i, j = only(celldofs(disc.dh_σ, c1)), only(celldofs(disc.dh_σ, c2))
        w = _facet_measure(disc, FacetIndex(c1, f1)) * abs(σ[i] - σ[j]) / 2
        J[c1] += w
        J[c2] += w
    end
    return J
end

"""
    dorfler_marking(η, θ)

Dörfler (bulk) marking: the smallest set of cells, taken in decreasing order of `η`, whose
indicators sum to at least `θ` times the total.
"""
function dorfler_marking(η::AbstractVector, θ::Real)
    0 < θ <= 1 || throw(ArgumentError("θ must be in (0, 1]"))
    total = sum(η)
    marked = Int[]
    total > 0 || return marked
    acc = zero(total)
    for c in sortperm(η; rev = true)
        push!(marked, c)
        acc += η[c]
        acc >= θ * total && break
    end
    return marked
end

"""
    transfer_conductivity(disc_old, σ_old, disc_new)

Conductivity coefficients on `disc_new` from `σ_old` on `disc_old` (e.g. after refinement or
coarsening): the L² projection of the old conductivity onto the new σ space, with the old
conductivity evaluated at the quadrature points of the new mesh. For piecewise constants this
is exact in both directions: refined cells inherit the value of their parent, coarsened cells
get the mean of their children.
"""
function transfer_conductivity(d_old::FerriteDiscretization, σ_old::AbstractVector, d_new::FerriteDiscretization)
    cv = d_new.cv_σ
    nq, nb = getnquadpoints(cv), getnbasefunctions(cv)
    points = Vec{Ferrite.getspatialdim(d_new.grid), Float64}[]
    for cell in CellIterator(d_new.dh_σ)
        reinit!(cv, cell)
        x = getcoordinates(cell)
        for q in 1:nq
            push!(points, spatial_coordinate(cv, q, x))
        end
    end
    ph = PointEvalHandler(d_old.grid, points)
    vals = evaluate_at_points(ph, d_old.dh_σ, σ_old, :σ)
    any(isnan, vals) && throw(ArgumentError("some points of the new mesh lie outside the old mesh"))
    b = zeros(ndofs(d_new.dh_σ))
    be = zeros(nb)
    p = 0
    for cell in CellIterator(d_new.dh_σ)
        reinit!(cv, cell)
        fill!(be, 0)
        for q in 1:nq
            p += 1
            v = vals[p] * getdetJdV(cv, q)
            for i in 1:nb
                be[i] += v * shape_value(cv, q, i)
            end
        end
        assemble!(b, celldofs(cell), be)
    end
    M = assemble_mass(d_new.dh_σ, cv)
    return cholesky(Symmetric(M)) \ b
end
