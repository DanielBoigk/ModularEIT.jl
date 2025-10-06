using Documenter
using ModularEIT

makedocs(
    sitename="ModularEIT.jl",
    modules=[ModularEIT],
)
deploydocs(
    repo="github.com/DanielBoigk/ModularEIT.jl.git",
)
