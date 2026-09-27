# Benchmark of the projected block CG solver on the EIT Neumann stiffness matrix
#     Lσ = ∫ σ ∇φᵢ⋅∇φⱼ dΩ   (Q1 elements on [-1,1]², cell-wise random σ, pure Neumann)
# on the CPU and on a CUDA GPU.
#
#   julia --project=benchmark -t auto benchmark/projected_block_cg.jl [sizes...]
#
# Writes a Markdown table to benchmark/projected_block_cg_results.md.

using ModularEIT
using Ferrite
using LinearAlgebra
using SparseArrays
using Random
using Printf
using CUDA
using CUDA.CUSPARSE

function assemble_neumann(n; contrast = 10.0, rng = MersenneTwister(0))
    grid = generate_grid(Quadrilateral, (n, n))
    ip = Lagrange{RefQuadrilateral, 1}()
    cv = CellValues(QuadratureRule{RefQuadrilateral}(2), ip)
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    K = allocate_matrix(dh)
    asm = start_assemble(K)
    Ke = zeros(4, 4)
    σ = 1 .+ (contrast - 1) .* rand(rng, getncells(grid))
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        s = σ[cellid(cell)]
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for i in 1:4, j in 1:4
                Ke[i, j] += s * (shape_gradient(cv, q, i) ⋅ shape_gradient(cv, q, j)) * dΩ
            end
        end
        assemble!(asm, celldofs(cell), Ke)
    end
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, union(getfacetset.(Ref(grid), ("left", "right", "top", "bottom"))...), x -> 0.0))
    close!(ch)
    return K, ch.prescribed_dofs
end

function boundary_rhs(ndofs, bd, s; rng = MersenneTwister(1))
    B = zeros(ndofs, s)
    B[bd, :] .= randn(rng, length(bd), s)
    B[bd, :] .-= sum(B[bd, :]; dims = 1) ./ length(bd)              # Σ bᵢ = 0 (compatible)
    return B
end

todev(x::SparseMatrixCSC) = CuSparseMatrixCSR(x)
todev(x) = CuArray(x)

# minimum wall time of `reps` runs (after one warm-up run)
function timeit(f; reps = 3)
    f()
    return minimum(@elapsed(f()) for _ in 1:reps)
end

function run_case(A, B, w, M; device, T, rtol)
    Ah, Bh = SparseMatrixCSC{T, Int}(A), T.(B)
    Ad, Bd, wd = device == :gpu ? (todev(Ah), todev(Bh), T.(w)) : (Ah, Bh, T.(w))
    X = fill!(similar(Bd), 0)
    ws = BlockCGWorkspace(Ad, Bd; grounding = wd)
    stats = Ref{Any}()
    run() = (fill!(X, 0); stats[] = pbcg!(X, ws, Ad, Bd; M, rtol); device == :gpu && CUDA.synchronize())
    t = timeit(run)
    relres = norm(Ah * Array(X) - Bh) / norm(Bh)
    return t, stats[].iterations, relres
end

function main(sizes = (128, 256, 512, 1024); s = 32)
    lines = String[]
    push!(lines, "| grid | dofs | device | precision | preconditioner | setup [s] | solve [s] | iterations | time/iteration [ms] | rel. residual |")
    push!(lines, "|:--|--:|:--|:--|:--|--:|--:|--:|--:|--:|")
    for n in sizes
        A, bd = assemble_neumann(n)
        N = size(A, 1)
        B = boundary_rhs(N, bd, s)
        w = boundary_grounding(N, bd)
        for device in (:cpu, :gpu), T in (Float64, Float32)
            device == :cpu && T == Float32 && continue
            rtol = T == Float64 ? 1e-8 : 1f-5
            for pre in (:none, :jacobi, :amg)
                pre == :none && n > 512 && continue                     # too slow to be useful
                device == :cpu && pre == :jacobi && n > 512 && continue
                Ah = SparseMatrixCSC{T, Int}(A)
                conv = device == :gpu ? todev : identity
                tsetup = @elapsed M = pre == :none ? nothing :
                                      pre == :jacobi ? JacobiPreconditioner(Ah; to_device = conv) :
                                                       AMGPreconditioner(Ah, s; to_device = conv)
                t, it, rr = run_case(A, B, w, M; device, T, rtol)
                line = @sprintf("| %d² | %d | %s | %s | %s | %.3f | %.3f | %d | %.2f | %.1e |",
                                n, N, device, T, pre, tsetup, t, it, 1e3 * t / max(it, 1), rr)
                println(line); flush(stdout)
                push!(lines, line)
            end
        end
    end
    header = """
    # Projected block CG benchmark

    EIT Neumann stiffness matrix (Q1, cell-wise random σ ∈ [1, 10]), $s right-hand sides with
    mean-zero random boundary currents, grounding Σ_boundary uᵢ = 0, rtol = 1e-8 (Float64) /
    1e-5 (Float32). CPU: $(Sys.cpu_info()[1].model) ($(Threads.nthreads()) Julia threads,
    $(BLAS.get_num_threads()) BLAS threads). GPU: $(CUDA.name(CUDA.device())).
    AMG setup (hierarchy construction) runs on the CPU in both cases.

    """
    out = get(ENV, "PBCG_RESULTS", joinpath(@__DIR__, "projected_block_cg_results.md"))
    open(out, "w") do io
        print(io, header, join(lines, "\n"), "\n")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? (128, 256, 512, 1024) : parse.(Int, ARGS))
end
