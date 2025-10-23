using Documenter
using ModularEIT

makedocs(
    sitename="ModularEIT.jl",
    modules=[ModularEIT],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ]
    checkdocs = :none,
)

deploydocs(
    repo="github.com/DanielBoigk/ModularEIT.jl.git",
)
