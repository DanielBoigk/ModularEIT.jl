# Benchmark of the projected sparse Cholesky solver (CHOLMOD on the CPU, cuDSS on the GPU),
# compared with the projected block CG solver + AMG, on the EIT Neumann stiffness matrix.
#
#   julia --project=benchmark -t auto benchmark/projected_cholesky.jl [sizes...]
#
# Writes benchmark/projected_cholesky_results.md.

using CUDSS
include("projected_block_cg.jl")

function cholesky_bench(sizes; s = 32)
    lines = ["| grid | dofs | device | precision | first factorisation [s] | refactorisation [s] | solve, $s RHS [s] | factor + solve [s] | block CG + AMG solve [s] | rel. residual |",
             "|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for n in sizes
        A, bd = assemble_neumann(n)
        A2, _ = assemble_neumann(n; rng = MersenneTwister(7))       # new σ, same pattern
        N = size(A, 1)
        w = boundary_grounding(N, bd)
        B = boundary_rhs(N, bd, s)
        for (device, T) in ((:cpu, Float64), (:gpu, Float64), (:gpu, Float32))
            conv = device == :gpu ? todev : identity
            sync() = device == :gpu && CUDA.synchronize()
            Ah, A2h = SparseMatrixCSC{T, Int}(A), SparseMatrixCSC{T, Int}(A2)
            Bd = conv(T.(B))
            X = similar(Bd)
            projected_cholesky(Ah; grounding = T.(w), nrhs = s, to_device = conv)   # compile
            tfact = @elapsed (F = projected_cholesky(Ah; grounding = T.(w), nrhs = s, to_device = conv); sync())
            trefac = timeit(() -> (refactor!(F, A2h); sync()))
            tsolve = timeit(() -> (ldiv!(X, F, Bd); sync()))
            rr = norm(A2h * Array(X) - T.(B)) / norm(B)
            # reference: projected block CG with AMG on the same device/precision (A2)
            M = AMGPreconditioner(A2h, s; to_device = conv)
            tcg = run_case(A2, B, w, M; device, T, rtol = T == Float64 ? 1e-8 : 1f-5)[1]
            line = @sprintf("| %d² | %d | %s | %s | %.3f | %.3f | %.4f | %.3f | %.3f | %.1e |",
                            n, N, device, T, tfact, trefac, tsolve, trefac + tsolve, tcg, rr)
            println(line); flush(stdout)
            push!(lines, line)
        end
    end
    header = """
    # Projected Cholesky benchmark

    EIT Neumann stiffness matrix (Q1, cell-wise random σ ∈ [1, 10]), $s mean-zero boundary
    current patterns, grounding Σ_boundary uᵢ = 0. CPU: CHOLMOD (Float64 only), GPU: cuDSS.
    *First factorisation* includes the symbolic analysis (ordering); *refactorisation* is the
    numeric factorisation for a new σ with the same pattern, which is what an EIT reconstruction
    repeats every iteration. The last timing column is the projected block CG + AMG solve (setup
    excluded, rtol 1e-8 / 1e-5) on the same device for comparison. Times are minima of 3 runs.
    CPU: $(Sys.cpu_info()[1].model) ($(Threads.nthreads()) threads), GPU: $(CUDA.name(CUDA.device())).

    """
    open(joinpath(@__DIR__, "projected_cholesky_results.md"), "w") do io
        print(io, header, join(lines, "\n"), "\n")
    end
end

cholesky_bench(isempty(ARGS) ? (32, 64, 128, 256, 512, 1024) : parse.(Int, ARGS))
