# Regrounding of measured voltages and SVD of boundary data pairs.
#
# Voltages are defined up to a constant per pattern; groundings differ by projections along 1,
# so data can be shifted between them exactly (`reground`). Linear combinations of measured
# pairs (Iₛ, Vₛ) are measured pairs of the same experiment, so the pairs can be rotated into
# orthonormal patterns ordered by their singular values (`pattern_svd`). Which patterns are
# "orthonormal" depends on the inner product: Euclidean (nodal values, fits the nodal-sum
# ground) or L²(Γ) (boundary mass matrix, fits the boundary-mean ground; mesh independent).

"""
    reground(V, w)
    reground(fm, V, grounding)

Shift every column of `V` by a constant so that `wᵀv = 0` (`V - 1 (wᵀV)/(wᵀ1)`). Groundings
differ only by such shifts, so data can be moved between them and back without loss.

With a forward model: `grounding = :integral` uses `fm.measure_weights` (zero boundary mean for
the continuum model), `:nodal` uses unit weights (zero sum of the measured values). For the
electrode models both are the plain mean over the measured electrodes.
"""
reground(V::AbstractVecOrMat, w::AbstractVector) = _project!(copy(V), w)

function reground(fm::ForwardModel, V::AbstractVecOrMat, grounding::Symbol)
    grounding === :integral && return reground(V, fm.measure_weights)
    grounding === :nodal && return reground(V, ones(size(V, 1)))
    throw(ArgumentError("grounding must be :integral or :nodal, got :$grounding"))
end

"""
    pattern_svd(disc, fm, currents, voltages; metric = :euclidean, noise = nothing, reference = nothing)
    pattern_svd(currents, voltages, Mi, Mv; noise = nothing, reference = nothing)

New pairs of boundary data from measured pairs `(currents, voltages)` (`n_inject × s`,
`n_measure × s`): `currents * C`, `voltages * C` for the `s × s` matrix `C` that makes the new
currents orthonormal in the inner product `Mi` and the new voltages orthogonal in `Mv`, sorted
by decreasing singular value (the norms of the new voltages). The voltages are first regrounded
consistently with the metric (`Mv`-weighted mean zero). Returns
`(; currents, voltages, values, Mi, Mv, combination, projection, noise, noise_levels, reference)`
with the combination matrix `C` (new pairs = old pairs · `C`) and the measurement modes
`projection` (one row per mode, orthonormal in the `Mv⁻¹` inner product, ordered like the
patterns): `projection * (voltages - reference)` is the diagonal matrix of `values`. For a
two-sided truncation of the data, keep the leading rows as a [`ProjectedMisfit`](@ref).

With a `reference` (the voltages of a reference conductivity for the same currents, or, with a
forward model, the reference conductivity itself), the SVD is taken of the difference
`voltages - reference` (Isaacson's distinguishability: the new currents are the patterns that
best distinguish the unknown from the reference), `values` are the singular values of the
difference, and the returned `voltages` and `reference` are the measured and reference data
rotated by the same `C`, so that the pairs remain valid data for any objective.

With a `noise` model of the measured voltages ([`GaussianNoise`](@ref),
[`RelativeGaussianNoise`](@ref); the currents are taken as exact), `noise` is the noise of the
new voltages, a [`GaussianNoise`](@ref) with one standard deviation per entry (the rotation
mixes patterns of different noise; use it for [`discrepancy_target`](@ref)), and `noise_levels`
the expected `Mv`-norm of the noise in every new pattern, the scale to compare `values` with (see
[`truncate_patterns`](@ref)). Both are `nothing` without a noise model.

For relative noise the absolute data have a signal-to-noise ratio of about `1/δ` in every
pattern, so truncation at the noise level removes nothing: it is the difference to a reference
whose singular values decay below the noise.

`metric`:
- `:euclidean`: `Mi = Mv = I` (nodal/electrode values; matches the nodal-sum ground),
- `:L2`: the L²(Γ) inner products, i.e. the boundary mass matrix for the continuum model
  (current densities and voltages), and for gap and complete electrode models `|e_ℓ|` for
  electrode voltages and `1/|e_ℓ|` for electrode currents (their densities `I_ℓ/|e_ℓ|`).
  Mesh independent; not defined for point electrodes.
- a tuple `(Mi, Mv)` of symmetric positive definite matrices.

On uniform boundary meshes the two built-in metrics give the same patterns up to scaling. For
the continuum model with the L² metric the values approximate the singular values of the
Neumann-to-Dirichlet map on the span of the input patterns.
"""
function pattern_svd(disc::AbstractDiscretization, fm::ForwardModel, currents::AbstractMatrix,
                     voltages::AbstractMatrix; metric = :euclidean, noise = nothing, reference = nothing)
    Mi, Mv = _pattern_metrics(disc, fm, metric)
    reference isa AbstractVector && (reference = forward_neumann(fm, reference, currents)[1])
    return pattern_svd(currents, voltages, Mi, Mv; noise, reference)
end

function pattern_svd(currents::AbstractMatrix, voltages::AbstractMatrix, Mi::AbstractMatrix, Mv::AbstractMatrix;
                     noise = nothing, reference = nothing)
    G, V = Matrix{Float64}(currents), Matrix{Float64}(voltages)
    size(G, 2) == size(V, 2) || throw(DimensionMismatch("currents and voltages need the same number of patterns"))
    size(Mi) == (size(G, 1), size(G, 1)) || throw(DimensionMismatch("Mi must be $(size(G, 1)) × $(size(G, 1))"))
    size(Mv) == (size(V, 1), size(V, 1)) || throw(DimensionMismatch("Mv must be $(size(V, 1)) × $(size(V, 1))"))
    w = Mv * ones(size(V, 1))
    Vg = reground(V, w)
    R = cholesky(Symmetric(G' * Mi * G)).U           # currents: Ĝ = G R⁻¹ is Mi-orthonormal
    Ref = nothing
    if reference !== nothing
        size(reference) == size(V) || throw(DimensionMismatch("the reference needs the size of the voltages"))
        Ref = reground(Matrix{Float64}(reference), w)
    end
    Ĝ = G / R
    Lv = cholesky(Symmetric(Matrix(Mv))).L           # ‖v‖²_Mv = ‖Lvᵀ v‖²
    F = svd(Lv' * ((Ref === nothing ? Vg : Vg - Ref) / R))
    C = R \ F.V
    out = (; currents = Ĝ * F.V, voltages = Vg * C, values = F.S, Mi, Mv, combination = C,
           projection = Matrix(F.U' * Lv'))
    ref = Ref === nothing ? nothing : Ref * C
    noise === nothing && return (; out..., noise = nothing, noise_levels = nothing, reference = ref)
    # rows of E C are independent with variances S C.², then the regrounding Π = I - 1wᵀ/(wᵀ1):
    # E ‖Lvᵀ Π (E c_k)‖² = Σᵢ ‖Lvᵀ Π eᵢ‖² Var[i, k]
    Var = (abs2.(_noise_std(noise, V)) .* ones(size(V))) * abs2.(C)
    B = Lv' * (I - ones(size(V, 1)) * w' ./ sum(w))
    levels = sqrt.(vec(vec(sum(abs2, B; dims = 1))' * Var))
    return (; out..., noise = GaussianNoise(sqrt.(Var)), noise_levels = levels, reference = ref)
end

"""
    truncate_patterns(p; τ = 2, measurements = nothing)
    truncate_patterns(p, K; measurements = nothing)

The leading pairs of a [`pattern_svd`](@ref) result `p`: the first `K`, or those before the first
pattern whose singular value drops to `τ` times its noise level (needs `pattern_svd(...; noise)`;
at least one pair is kept).
The trailing pairs mostly carry noise; discarding them regularises in data space, without an
assumption on the conductivity. Directions that carry only noise do not have singular values
below the noise level but at it (the SVD of noisy data has a noise floor), so `τ` must lie
clearly above 1. Returns a named tuple with the same fields, restricted to the
retained pairs. With `measurements = M`, only the leading `M` measurement modes are kept in
`projection`; use them as `misfit = ProjectedMisfit(t.projection)` for `K M` residuals.
"""
function truncate_patterns(p::NamedTuple; τ::Real = 2, measurements = nothing)
    p.noise_levels === nothing &&
        throw(ArgumentError("no noise levels: compute the pattern SVD with a noise model (pattern_svd(...; noise))"))
    k = findfirst(p.values .<= τ .* p.noise_levels)
    return truncate_patterns(p, k === nothing ? length(p.values) : max(k - 1, 1); measurements)
end

function truncate_patterns(p::NamedTuple, K::Integer; measurements = nothing)
    1 <= K <= length(p.values) || throw(ArgumentError("K must lie in 1:$(length(p.values)), got $K"))
    M = measurements === nothing ? size(p.projection, 1) : Int(measurements)
    1 <= M <= size(p.projection, 1) || throw(ArgumentError("measurements must lie in 1:$(size(p.projection, 1)), got $M"))
    r = 1:K
    return (; currents = p.currents[:, r], voltages = p.voltages[:, r], values = p.values[r], Mi = p.Mi, Mv = p.Mv,
            combination = p.combination[:, r], projection = p.projection[1:M, :],
            noise = p.noise === nothing ? nothing : GaussianNoise(p.noise.std[:, r]),
            noise_levels = p.noise_levels === nothing ? nothing : p.noise_levels[r],
            reference = p.reference === nothing ? nothing : p.reference[:, r])
end

function _pattern_metrics(disc::AbstractDiscretization, fm::ForwardModel, metric)
    metric isa Tuple && length(metric) == 2 && return metric
    ni, nm = n_inject(fm), n_measure(fm)
    metric === :euclidean && return (Matrix(1.0I, ni, ni), Matrix(1.0I, nm, nm))
    metric === :L2 || throw(ArgumentError("metric must be :euclidean, :L2 or a tuple (Mi, Mv), got $metric"))
    model = fm.model
    if model isa ContinuumModel
        M = Matrix(fm.Q * fm.P)                      # boundary mass matrix on the boundary dofs
        return (M, M)
    elseif model isa GapModel
        li = [electrode_length(disc, e) for e in model.inject]
        lm = [electrode_length(disc, e) for e in model.measure]
        return (Diagonal(1 ./ li), Diagonal(lm))
    elseif model isa CompleteElectrodeModel
        l = [electrode_length(disc, e) for e in model.electrodes]
        return (Diagonal(1 ./ l), Diagonal(l[model.measure]))
    end
    throw(ArgumentError("the L² metric is not defined for $(nameof(typeof(model))) (use :euclidean)"))
end
