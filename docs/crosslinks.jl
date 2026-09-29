# Checks the links between the API documentation and the theory wiki against the built sites:
#   docs/src/**.md  → https://…/dev/wiki/<folder>/<Article>    must exist as build/wiki/…html
#   markdown/**.md  → https://…/dev/api/<page>/#<anchor>       must be a docstring on that page
# Both sites are published under one URL (the wiki in `wiki/` next to the API docs), so the links
# are absolute and point to the deployed `dev` version.

const SITE = "https://danielboigk.github.io/ModularEIT.jl/dev/"

markdown_files(dir) = [joinpath(r, f) for (r, _, fs) in walkdir(dir) for f in fs if endswith(f, ".md")]

links_in(files, prefix) =
    [(f, m.captures[1]) for f in files for m in eachmatch(Regex("\\]\\(" * prefix * "([^)\\s]*)\\)"), read(f, String))]

# docs → wiki: needs the rendered wiki in build/wiki
function check_wiki_links(docs_src, wiki_build)
    bad = String[]
    for (f, path) in links_in(markdown_files(docs_src), SITE * "wiki/")
        target = isempty(path) || endswith(path, "/") ? joinpath(wiki_build, path, "index.html") :
                 joinpath(wiki_build, path * ".html")
        isfile(target) || push!(bad, "$(relpath(f)): wiki page `$path` does not exist")
    end
    return bad
end

# wiki → docs: the anchor must be the id of a docstring on the built API page
function check_api_links(vault, docs_build)
    bad = String[]
    ids = Dict{String, Set{String}}()
    for (f, link) in links_in(markdown_files(vault), SITE * "api/")
        page, anchor = occursin('#', link) ? split(link, '#'; limit = 2) : (link, "")
        page = rstrip(page, '/')
        html = get!(ids, page) do
            candidates = (joinpath(docs_build, "api", page, "index.html"), joinpath(docs_build, "api", page * ".html"))
            i = findfirst(isfile, candidates)
            i === nothing ? Set{String}() : Set(m.captures[1] for m in eachmatch(r"id=\"([^\"]+)\"", read(candidates[i], String)))
        end
        if isempty(html)
            push!(bad, "$(relpath(f)): API page `$page` does not exist")
        elseif !isempty(anchor) && !(anchor in html)
            push!(bad, "$(relpath(f)): no docstring `$anchor` on API page `$page`")
        end
    end
    return bad
end

function check_crosslinks(; docs_src, docs_build, vault, wiki_build = nothing)
    bad = check_api_links(vault, docs_build)
    wiki_build === nothing || append!(bad, check_wiki_links(docs_src, wiki_build))
    isempty(bad) || error("broken links between the API docs and the wiki:\n  " * join(bad, "\n  "))
    @info "Links between the API docs and the wiki are valid" (wiki_build === nothing ? "(wiki not built: docs → wiki unchecked)" : "")
    return nothing
end
