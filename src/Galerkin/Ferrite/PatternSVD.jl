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
    pattern_svd(disc, fm, currents, voltages; metric = :euclidean)
    pattern_svd(currents, voltages, Mi, Mv)

New pairs of boundary data from measured pairs `(currents, voltages)` (`n_inject × s`,
`n_measure × s`): `currents * C`, `voltages * C` for the `s × s` matrix `C` that makes the new
currents orthonormal in the inner product `Mi` and the new voltages orthogonal in `Mv`, sorted
by decreasing singular value (the norms of the new voltages). The voltages are first regrounded
consistently with the metric (`Mv`-weighted mean zero). Returns
`(; currents, voltages, values, Mi, Mv)`.

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
function pattern_svd(disc::FerriteDiscretization, fm::ForwardModel, currents::AbstractMatrix,
                     voltages::AbstractMatrix; metric = :euclidean)
    Mi, Mv = _pattern_metrics(disc, fm, metric)
    return pattern_svd(currents, voltages, Mi, Mv)
end

function pattern_svd(currents::AbstractMatrix, voltages::AbstractMatrix, Mi::AbstractMatrix, Mv::AbstractMatrix)
    G, V = Matrix{Float64}(currents), Matrix{Float64}(voltages)
    size(G, 2) == size(V, 2) || throw(DimensionMismatch("currents and voltages need the same number of patterns"))
    size(Mi) == (size(G, 1), size(G, 1)) || throw(DimensionMismatch("Mi must be $(size(G, 1)) × $(size(G, 1))"))
    size(Mv) == (size(V, 1), size(V, 1)) || throw(DimensionMismatch("Mv must be $(size(V, 1)) × $(size(V, 1))"))
    Vg = reground(V, Mv * ones(size(V, 1)))
    R = cholesky(Symmetric(G' * Mi * G)).U           # currents: Ĝ = G R⁻¹ is Mi-orthonormal
    Ĝ, V̂ = G / R, Vg / R
    Lv = cholesky(Symmetric(Matrix(Mv))).L           # ‖v‖²_Mv = ‖Lvᵀ v‖²
    F = svd(Lv' * V̂)
    return (; currents = Ĝ * F.V, voltages = V̂ * F.V, values = F.S, Mi, Mv)
end

function _pattern_metrics(disc::FerriteDiscretization, fm::ForwardModel, metric)
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
