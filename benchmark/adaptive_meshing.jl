# Uniform vs adaptive refinement for the EIT forward problem.
#
# Square [-1, 1]² (pixel-image domain), complete electrode model with 16 electrodes (4 per side,
# width 0.25, centred at ±0.25 and ±0.75, so half the boundary is covered and every electrode
# edge lies on a mesh line of all meshes from 16 × 16 on: the electrode geometry is identical on
# every mesh), small contact impedance so the electrode edges are sharp, piecewise constant
# conductivity with a conductive and a resistive inclusion, sampled at the cell centroids of each
# mesh. Quantity of interest: the electrode voltages of 16 trigonometric current patterns,
# compared with a uniform 1024 × 1024 reference. Adaptive meshes start from 16 × 16 cells and
# may refine 6 levels, i.e. never below the reference cell size.
#
# Strategies (adaptive ones: Dörfler marking with θ = 0.3; cells at the maximum level are
# excluded from marking):
#   uniform    refine all cells
#   ZZ         flux-recovery (Zienkiewicz–Zhu) indicator: energy-norm error of the states
#   RES        residual indicator (element residuals, current-density jumps, boundary residuals)
#   GO         goal-oriented indicator: ZZ error of the states × ZZ error of the measurement duals
#   GO-res     goal-oriented indicator from residual estimates of states and duals
#   σ-jump     conductivity jump indicator (what a reconstruction-driven refinement would do;
#              blind to the electrodes)
#   GO+σ-jump  sum of the normalised GO and σ-jump indicators
#
# Run: SCENARIO=aligned STRATEGIES=uniform,GO-res julia --project benchmark/adaptive_meshing.jl
#      (prints a table; STRATEGIES defaults to all)
using ModularEIT, ModularEITFerrite, Ferrite, LinearAlgebra, Printf

const L = 16
const Z = 1e-3
# SCENARIO = "circles": round inclusions, sampled at the centroids of each mesh (the discrete
#   inclusion shape changes with the mesh). SCENARIO = "aligned": square inclusions that are
#   unions of cells of the 16 × 16 base mesh, so σ is represented exactly on every mesh and only
#   the discretisation error of the potential remains.
const SCENARIO = get(ENV, "SCENARIO", "circles")
inbox(x, a, b, c, d) = a < x[1] < b && c < x[2] < d
σfun(x) = SCENARIO == "aligned" ?
    (inbox(x, 0.125, 0.625, -0.125, 0.375) ? 5.0 : inbox(x, -0.625, -0.375, -0.5, -0.25) ? 0.2 : 1.0) :
    (hypot(x[1] - 0.3, x[2] - 0.2) < 0.35 ? 5.0 : hypot(x[1] + 0.4, x[2] + 0.35) < 0.2 ? 0.2 : 1.0)

# electrodes of width 0.25 centred at ±0.25, ±0.75 on every side, counterclockwise from (1, -0.75)
function square_electrodes(disc)
    centres = (-0.75, -0.25, 0.25, 0.75)
    sides = ((x -> x[1] ≈ 1, x -> x[2]), (x -> x[2] ≈ 1, x -> -x[1]),
             (x -> x[1] ≈ -1, x -> -x[2]), (x -> x[2] ≈ -1, x -> x[1]))
    els = [FacetIndex[] for _ in 1:L]
    for f in disc.boundary_facets
        m = ModularEITFerrite._facet_midpoint(disc.grid, f)
        for (k, (onside, t)) in enumerate(sides), (j, c) in enumerate(centres)
            onside(m) && abs(t(m) - c) < 0.125 && push!(els[4(k - 1) + j], f)
        end
    end
    return els
end

function solve_on(grid)
    t = @elapsed begin
        disc = FerriteDiscretization(grid)
        fm = ForwardModel(disc, CompleteElectrodeModel(square_electrodes(disc), Z))
        σ = interpolate_function(disc, σfun)
        I = trigonometric_patterns(fm, L ÷ 2)
        U, X = forward_neumann(fm, σ, I)
    end
    return (; disc, fm, σ, I, U, X, n = fm.n, t)
end

relerr(U, Uref) = norm(U - Uref) / norm(Uref)

println("reference: uniform 1024 × 1024 ...")
ref = solve_on(generate_grid(Quadrilateral, (1024, 1024)))
Uref = ref.U
@printf("reference: %d unknowns, %.1f s\n\n", ref.n, ref.t)

rows = Tuple{String, Int, Float64, Float64, Float64, Float64}[]

const STRATEGIES = split(get(ENV, "STRATEGIES", "uniform,ZZ,RES,GO,GO-res,σ-jump,GO+σ-jump"), ",")

"uniform" in STRATEGIES && for m in (16, 32, 64, 128, 256, 512)
    r = solve_on(generate_grid(Quadrilateral, (m, m)))
    η = sqrt(sum(flux_recovery_indicator(r.disc, r.σ, r.X)))
    push!(rows, ("uniform", r.n, relerr(r.U, Uref), η, r.t, 0.0))
end

zz(r) = flux_recovery_indicator(r.disc, r.σ, r.X)
res(r) = residual_indicator(r.disc, r.fm, r.σ, r.X, r.I)
go(r) = goal_oriented_indicator(r.disc, r.fm, r.σ, r.X)
gores(r) = goal_oriented_indicator(r.disc, r.fm, r.σ, r.X; estimator = :residual, currents = r.I)
jump(r) = jump_indicator(r.disc, r.σ)
normalised(η) = sum(η) > 0 ? η ./ sum(η) : η

for (name, indicator) in (("ZZ", zz), ("RES", res), ("GO", go), ("GO-res", gores), ("σ-jump", jump),
                          ("GO+σ-jump", r -> normalised(go(r)) .+ normalised(jump(r))))
    name in STRATEGIES || continue
    am = AdaptiveMesh(generate_grid(Quadrilateral, (16, 16)); maxlevel = 6)
    for step in 1:60
        r = solve_on(current_grid(am))
        ηzz = sqrt(sum(zz(r)))
        ti = @elapsed (η = indicator(r))
        push!(rows, (name, r.n, relerr(r.U, Uref), ηzz, r.t, ti))
        r.n > 300_000 && break
        η[cell_levels(am) .>= max_level(am)] .= 0          # cannot be refined further
        marked = dorfler_marking(η, 0.3)
        isempty(marked) && break
        refine_mesh!(am, marked)
    end
end

println("| strategy | unknowns | rel. voltage error | ZZ estimate | solve [s] | indicator [s] |")
println("|:--|--:|--:|--:|--:|--:|")
for (name, n, e, η, t, ti) in rows
    @printf("| %s | %d | %.2e | %.2e | %.2f | %.2f |\n", name, n, e, η, t, ti)
end
