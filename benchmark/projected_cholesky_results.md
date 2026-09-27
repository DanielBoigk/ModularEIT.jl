# Projected Cholesky benchmark

EIT Neumann stiffness matrix (Q1, cell-wise random σ ∈ [1, 10]), 32 mean-zero boundary
current patterns, grounding Σ_boundary uᵢ = 0. CPU: CHOLMOD (Float64 only), GPU: cuDSS.
*First factorisation* includes the symbolic analysis (ordering); *refactorisation* is the
numeric factorisation for a new σ with the same pattern, which is what an EIT reconstruction
repeats every iteration. The last timing column is the projected block CG + AMG solve (setup
excluded, rtol 1e-8 / 1e-5) on the same device for comparison. Times are minima of 3 runs.
CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 threads), GPU: NVIDIA GeForce RTX 3080.

| grid | dofs | device | precision | first factorisation [s] | refactorisation [s] | solve, 32 RHS [s] | factor + solve [s] | block CG + AMG solve [s] | rel. residual |
|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|
| 32² | 1089 | cpu | Float64 | 0.001 | 0.000 | 0.0004 | 0.001 | 0.013 | 1.4e-14 |
| 32² | 1089 | gpu | Float64 | 0.049 | 0.000 | 0.0003 | 0.001 | 0.008 | 1.0e-14 |
| 32² | 1089 | gpu | Float32 | 0.012 | 0.000 | 0.0003 | 0.000 | 0.004 | 5.6e-06 |
| 64² | 4225 | cpu | Float64 | 0.005 | 0.001 | 0.0026 | 0.004 | 0.049 | 7.5e-14 |
| 64² | 4225 | gpu | Float64 | 0.023 | 0.001 | 0.0006 | 0.001 | 0.013 | 8.7e-14 |
| 64² | 4225 | gpu | Float32 | 0.027 | 0.000 | 0.0004 | 0.001 | 0.004 | 1.3e-05 |
| 128² | 16641 | cpu | Float64 | 0.017 | 0.007 | 0.0113 | 0.018 | 0.208 | 1.9e-13 |
| 128² | 16641 | gpu | Float64 | 0.060 | 0.002 | 0.0012 | 0.003 | 0.034 | 1.8e-13 |
| 128² | 16641 | gpu | Float32 | 0.056 | 0.001 | 0.0006 | 0.002 | 0.006 | 2.1e-05 |
| 256² | 66049 | cpu | Float64 | 0.113 | 0.024 | 0.0756 | 0.100 | 0.865 | 1.4e-12 |
| 256² | 66049 | gpu | Float64 | 0.203 | 0.006 | 0.0044 | 0.011 | 0.094 | 1.4e-12 |
| 256² | 66049 | gpu | Float32 | 0.181 | 0.003 | 0.0022 | 0.006 | 0.016 | 1.2e-04 |
| 512² | 263169 | cpu | Float64 | 0.297 | 0.138 | 0.3521 | 0.490 | 4.600 | 5.0e-12 |
| 512² | 263169 | gpu | Float64 | 0.870 | 0.026 | 0.0168 | 0.043 | 0.418 | 5.0e-12 |
| 512² | 263169 | gpu | Float32 | 0.841 | 0.014 | 0.0088 | 0.022 | 0.053 | 1.2e-03 |
| 1024² | 1050625 | cpu | Float64 | 1.464 | 0.805 | 1.4105 | 2.215 | 20.459 | 2.9e-11 |
| 1024² | 1050625 | gpu | Float64 | 3.785 | 0.126 | 0.0658 | 0.192 | 1.769 | 3.0e-11 |
| 1024² | 1050625 | gpu | Float32 | 3.658 | 0.061 | 0.0345 | 0.095 | 0.243 | 7.9e-03 |
