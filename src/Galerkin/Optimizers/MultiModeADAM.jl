using Optimisers

struct MultiMode{R <: Optimisers.AbstractRule} <: Optimisers.AbstractRule
    modes::Dict{Int, Tuple{Bool, R}}  # Key: rule index, Value: (is_ready, rule)
end

function MultiMode(rules::Vector{<:Optimisers.AbstractRule})
    modes = Dict{Int, Tuple{Bool, eltype(rules)}}()
    for (i, rule) in enumerate(rules)
        modes[i] = (true, rule)  # Initially, all rules are ready
    end
    return MultiMode{eltype(rules)}(modes)
end

function Optimisers.initial_state(m::MultiMode, x)
    # Initialize state for each rule
    states = Dict{Int, Any}()
    for (i, (ready, rule)) in m.modes
        states[i] = Optimisers.initial_state(rule, x)
    end
    return states
end

function Optimisers.apply!(m::MultiMode, state, x, Δ)
    for (i, (ready, rule)) in m.modes
        if ready
            # Apply the rule's update
            x, state[i] = Optimisers.apply!(rule, state[i], x, Δ)
            # Mark as not ready after applying
            m.modes[i] = (false, rule)
        end
    end
    return x, state
end
