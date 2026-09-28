# Newest vertex bisection (NVB) of linear triangle meshes: conforming local refinement without
# hanging nodes, so every finite element space (also continuous σ) works on the refined mesh.
#
# Every triangle is stored as (i, j, k), counter-clockwise, with refinement edge (i, j) and
# newest vertex k. Bisection inserts the midpoint m of (i, j) and creates the children
# (k, i, m) and (j, k, m): both counter-clockwise, with newest vertex m and refinement edges
# (k, i) and (j, k). The initial refinement edge of every triangle is its longest edge.
#
# Conformity: an edge that has been split (its midpoint exists) must not remain in any leaf.
# The closure repeatedly bisects every leaf that still contains a split edge; a leaf whose split
# edge is not its refinement edge is first bisected along its refinement edge, and the child that
# inherits the split edge has it as refinement edge, so it is bisected in the next round.

mutable struct BisectionMesh
    nodes::Vector{Vec{2, Float64}}
    tris::Vector{NTuple{3, Int}}                 # leaves (i, j, k)
    levels::Vector{Int}                          # number of bisections from the initial cell
    cellsets::Vector{Vector{String}}             # cell sets of every leaf
    edgesets::Dict{String, Set{Tuple{Int, Int}}} # facet sets as sorted node pairs
    midpoints::Dict{Tuple{Int, Int}, Int}        # split edge → midpoint node
    maxlevel::Int
end

_signed_area(p, q, r) = ((q - p)[1] * (r - p)[2] - (q - p)[2] * (r - p)[1]) / 2

function BisectionMesh(grid::Ferrite.AbstractGrid, maxlevel::Integer)
    nodes = [get_node_coordinate(grid, i) for i in 1:getnnodes(grid)]
    tris = NTuple{3, Int}[]
    for c in getcells(grid)
        a, b, cc = c.nodes
        _signed_area(nodes[a], nodes[b], nodes[cc]) < 0 && ((b, cc) = (cc, b))
        t = (a, b, cc)
        # rotate the longest edge to (i, j)
        lens = [norm(nodes[t[mod1(r + 1, 3)]] - nodes[t[r]]) for r in 1:3]
        r = argmax(lens)
        push!(tris, (t[r], t[mod1(r + 1, 3)], t[mod1(r + 2, 3)]))
    end
    names = [String[] for _ in tris]
    for (name, set) in grid.cellsets, c in set
        push!(names[c], name)
    end
    edgesets = Dict{String, Set{Tuple{Int, Int}}}()
    for (name, set) in grid.facetsets
        edgesets[name] = Set(minmax(Ferrite.facets(getcells(grid, fi.idx[1]))[fi.idx[2]]...) for fi in set)
    end
    return BisectionMesh(nodes, tris, zeros(Int, length(tris)), names, edgesets,
                         Dict{Tuple{Int, Int}, Int}(), maxlevel)
end

function _bisect!(m::BisectionMesh, t::Integer)
    i, j, k = m.tris[t]
    e = minmax(i, j)
    mid = get(m.midpoints, e, 0)
    if mid == 0
        push!(m.nodes, (m.nodes[i] + m.nodes[j]) / 2)
        mid = length(m.nodes)
        m.midpoints[e] = mid
        for set in values(m.edgesets)
            if e in set
                delete!(set, e)
                push!(set, minmax(e[1], mid), minmax(mid, e[2]))
            end
        end
    end
    l = m.levels[t] + 1
    m.tris[t] = (k, i, mid)
    push!(m.tris, (j, k, mid))
    m.levels[t] = l
    push!(m.levels, l)
    push!(m.cellsets, m.cellsets[t])
    return m
end

_has_split_edge(m::BisectionMesh, (i, j, k)) =
    haskey(m.midpoints, minmax(i, j)) || haskey(m.midpoints, minmax(j, k)) || haskey(m.midpoints, minmax(k, i))

# refine the marked leaves (those below the maximum level) and close the mesh
function _refine!(m::BisectionMesh, cells)
    todo = unique(c for c in cells if m.levels[c] < m.maxlevel)
    rounds = 0
    while !isempty(todo)
        (rounds += 1) > 10_000 && error("newest vertex bisection did not terminate")
        for t in todo
            _bisect!(m, t)
        end
        todo = [t for t in eachindex(m.tris) if _has_split_edge(m, m.tris[t])]
    end
    return m
end

function _creategrid(m::BisectionMesh)
    cells = [Triangle(t) for t in m.tris]
    nodes = [Node(x) for x in m.nodes]
    edge_names = Dict{Tuple{Int, Int}, Vector{String}}()
    for (name, set) in m.edgesets, e in set
        push!(get!(edge_names, e, String[]), name)
    end
    facetsets = Dict{String, Ferrite.OrderedSet{FacetIndex}}(n => Ferrite.OrderedSet{FacetIndex}() for n in keys(m.edgesets))
    for (c, t) in enumerate(m.tris), (lf, (a, b)) in enumerate(((t[1], t[2]), (t[2], t[3]), (t[3], t[1])))
        for name in get(edge_names, minmax(a, b), ())
            push!(facetsets[name], FacetIndex(c, lf))
        end
    end
    cellsets = Dict{String, Ferrite.OrderedSet{Int}}()
    for (c, names) in enumerate(m.cellsets), n in names
        push!(get!(cellsets, n, Ferrite.OrderedSet{Int}()), c)
    end
    return Grid(cells, nodes; facetsets, cellsets)
end
