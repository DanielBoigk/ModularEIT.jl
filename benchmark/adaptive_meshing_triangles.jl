# Uniform vs adaptive refinement on triangle meshes (newest vertex bisection).
#
# Unit disc (test/data/circle.msh, 2972 triangles), complete electrode model with 16 electrodes
# (half the boundary covered, contact impedance 1e-3). Electrodes are named facet sets of the
# base mesh and σ is a piecewise constant on the base mesh (two inclusions); both are carried
# exactly through bisection (facet sets are split, children inherit σ), so every mesh describes
# the same physical problem. Quantity of interest: electrode voltages of 16 trigonometric
# patterns, compared with the base mesh bisected uniformly 8 times.
#
# Run: julia --project benchmark/adaptive_meshing_triangles.jl
using ModularEIT, Ferrite, FerriteGmsh, LinearAlgebra, Printf

const L = 16
const Z = 1e-3
const MAXLEVEL = 8

base = redirect_stdout(devnull) do
    togrid(joinpath(@__DIR__, "..", "test", "data", "circle.msh"))
end
d0 = FerriteDiscretization(base)
for (ℓ, e) in enumerate(angular_electrodes(d0, L; coverage = 0.5))
    addfacetset!(base, "e$ℓ", Set(e))
end
d0 = FerriteDiscretization(base)
σfun(x) = hypot(x[1] - 0.3, x[2] - 0.2) < 0.35 ? 5.0 : hypot(x[1] + 0.4, x[2] + 0.35) < 0.2 ? 0.2 : 1.0
σ0 = interpolate_function(d0, σfun)

function solve_on(grid)
    t = @elapsed begin
        disc = FerriteDiscretization(grid)
        els = [collect(getfacetset(grid, "e$ℓ")) for ℓ in 1:L]
        fm = ForwardModel(disc, CompleteElectrodeModel(els, Z))
        σ = transfer_conductivity(d0, σ0, disc)
        I = trigonometric_patterns(fm, L ÷ 2)
        U, X = forward_neumann(fm, σ, I)
    end
    return (; disc, fm, σ, I, U, X, n = fm.n, t)
end
relerr(U, Uref) = norm(U - Uref) / norm(Uref)

println("reference: base mesh bisected uniformly $MAXLEVEL times ...")
amr = AdaptiveMesh(base; maxlevel = MAXLEVEL)
for _ in 1:MAXLEVEL
    refine_mesh!(amr)
end
ref = solve_on(current_grid(amr))
Uref = ref.U
@printf("reference: %d unknowns, %.1f s\n\n", ref.n, ref.t)

rows = Tuple{String, Int, Float64, Float64, Float64}[]
amu = AdaptiveMesh(base; maxlevel = MAXLEVEL)
for k in 0:MAXLEVEL-2
    r = solve_on(current_grid(amu))
    push!(rows, ("uniform", r.n, relerr(r.U, Uref), r.t, 0.0))
    refine_mesh!(amu)
end

res(r) = residual_indicator(r.disc, r.fm, r.σ, r.X, r.I)
gores(r) = goal_oriented_indicator(r.disc, r.fm, r.σ, r.X; estimator = :residual, currents = r.I)
for (name, indicator) in (("RES", res), ("GO-res", gores))
    am = AdaptiveMesh(base; maxlevel = MAXLEVEL)
    for step in 1:60
        r = solve_on(current_grid(am))
        ti = @elapsed (η = indicator(r))
        push!(rows, (name, r.n, relerr(r.U, Uref), r.t, ti))
        r.n > 200_000 && break
        η[cell_levels(am) .>= max_level(am)] .= 0
        marked = dorfler_marking(η, 0.3)
        isempty(marked) && break
        refine_mesh!(am, marked)
    end
end

println("| strategy | unknowns | rel. voltage error | solve [s] | indicator [s] |")
println("|:--|--:|--:|--:|--:|")
for (name, n, e, t, ti) in rows
    @printf("| %s | %d | %.2e | %.2f | %.2f |\n", name, n, e, t, ti)
end
