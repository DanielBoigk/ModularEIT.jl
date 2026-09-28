using Documenter
using ModularEIT

DocMeta.setdocmeta!(ModularEIT, :DocTestSetup, :(using ModularEIT); recursive=true)

makedocs(
    sitename="ModularEIT.jl",
    modules=[ModularEIT],
    authors="Daniel Boigk",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => [
            "Discretization" => "api/discretization.md",
            "Electrode Models & Forward Problem" => "api/forward.md",
            "Objectives" => "api/objectives.md",
            "Adaptive Meshing" => "api/adaptivity.md",
            "Linear Solvers" => "api/linear_solvers.md",
        ],
    ],
)

# The theory wiki (Obsidian vault in `markdown/`) is rendered with Quartz from `site/`
# into `build/wiki/`, so it is deployed together with the API docs.
# Set BUILD_WIKI=false to skip this step (e.g. when Node.js is not installed).
if get(ENV, "BUILD_WIKI", "true") != "false"
    site = joinpath(@__DIR__, "..", "site")
    isdir(joinpath(site, "node_modules")) || run(Cmd(`npm ci`; dir=site))
    run(Cmd(`npx quartz build -d ../markdown -o $(joinpath(@__DIR__, "build", "wiki"))`; dir=site))
end

deploydocs(
    repo="github.com/DanielBoigk/ModularEIT.jl.git",
)
